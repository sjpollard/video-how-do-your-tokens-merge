# Based on: https://github.com/facebookresearch/ToMe/blob/main/tome/patch/timm.py

import copy

import torch
import torch.nn.functional as F
from slowfast.models.videomae_video_model_builder import VideoMAE, Block, Attention

from tome.merge import bipartite_soft_matching, bipartite_soft_matching_drop, bipartite_soft_matching_hybrid, merge_source, merge_wavg
from tome.utils import parse_r


class ToMeBlock(Block):
    def forward(self, x):
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        if self.gamma_1 is None:
            attn, metric = self.attn(self.norm1(x), attn_size, self._tome_info["head_aggregation"])
            x = x + self.drop_path(attn)

            x = self.reduction_function(metric, x, self._tome_info)

            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            attn, metric = self.attn(self.norm1(x), attn_size, self._tome_info["head_aggregation"])
            x = x + self.drop_path(self.gamma_1 * attn)

            x = self.reduction_function(metric, x, self._tome_info)

            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x
    

class ToMeDuplicateBlock(ToMeBlock):
    def forward(self, x):
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        if self.gamma_1 is None:
            attn, metric = self.attn(self.norm1(x), attn_size, self._tome_info["head_aggregation"])

            x = self.reduction_function(metric, x, self._tome_info)
        else:
            attn, metric = self.attn(self.norm1(x), attn_size, self._tome_info["head_aggregation"])

            x = self.reduction_function(metric, x, self._tome_info)
        return x


class ToMeAttention(Attention):
    def forward(self, x, size: torch.Tensor = None, head_aggregation: str = 'mean'):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        ## Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        if head_aggregation == 'mean':
            metric = k.mean(1)
        elif head_aggregation == 'concat':
            metric = torch.cat(k.split(1, dim=1), dim=-1).squeeze(dim=1)

        return x, metric


def videomae_merge(metric, x, _tome_info):
    r = _tome_info["r"].pop(0)
    if r > 0:
        # Apply ToMe here
        merge, _ = bipartite_soft_matching(
            metric,
            r,
            _tome_info["class_token"],
            _tome_info["distill_token"],
            _tome_info["mode"]
        )
        if _tome_info["trace_source"]:
            _tome_info["source"] = merge_source(
                merge, x, _tome_info["source"]
            )
        pre_merge = x.size(1)
        x, _tome_info["size"] = merge_wavg(merge, x, _tome_info["size"])
        if _tome_info['verbose']:
            print(f'Merged {pre_merge} to {x.size(1)} tokens')
    
    return x


def videomae_drop(metric, x, _tome_info):
    r = _tome_info["r"].pop(0)
    if r > 0:
        # Apply ToMe here
        drop = bipartite_soft_matching_drop(
            metric,
            r,
            _tome_info["class_token"],
            _tome_info["distill_token"],
            _tome_info["mode"]
        )
        if _tome_info["trace_source"]:
            if _tome_info["source"] is None:
                n, t, _ = x.shape
                _tome_info["source"] = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)
            _tome_info["source"] = drop(
                _tome_info["source"]
            )
        pre_drop = x.size(1)
        x = drop(x)
        _tome_info["size"] = torch.ones((x.size(0), x.size(1), 1), device=x.device)
        if _tome_info['verbose']:
            print(f'Dropped {pre_drop} to {x.size(1)} tokens')
    
    return x


def videomae_hybrid(metric, x, _tome_info):
    r = _tome_info["r"].pop(0)
    if r > 0:
        # Apply ToMe here
        merge, _ = bipartite_soft_matching_hybrid(
            metric,
            r,
            _tome_info["class_token"],
            _tome_info["distill_token"],
            _tome_info["mode"],
            _tome_info["threshold"]
        )
        if _tome_info["trace_source"]:
            _tome_info["source"] = merge_source(
                merge, x, _tome_info["source"]
            )
        pre_merge = x.size(1)
        x, _tome_info["size"] = merge_wavg(merge, x, _tome_info["size"])
        if _tome_info['verbose']:
            print(f'Merged {pre_merge} to {x.size(1)} tokens')
    
    return x


def apply_duplicate_patch(model, layer_to_duplicate, quantity):
    for i in range(layer_to_duplicate, layer_to_duplicate + quantity - 1):
        model.model.blocks.insert(index=i, module=copy.deepcopy(model.model.blocks[i]))
        model.model.blocks[i].__class__ = ToMeDuplicateBlock


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.model.blocks), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().forward(*args, **kwdargs)

    return ToMeVisionTransformer


def apply_patch(
    model_wrapper: VideoMAE, trace_source: bool = False, prop_attn: bool = False, mode: str = 'merge',
    head_aggregation: str = 'mean', threshold: float = 0.0, verbose: bool = False
):  
    model = model_wrapper.model
    ToMeVisionTransformer = make_tome_class(model_wrapper.__class__)

    model_wrapper.__class__ = ToMeVisionTransformer
    model_wrapper.r = 0
    model_wrapper._tome_info = {
        "r": model_wrapper.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "verbose": verbose,
        "class_token": False,
        "distill_token": False,
        "mode": mode,
        "head_aggregation": head_aggregation,
        "threshold": threshold
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    if mode in ['merge', 'random_merge']:
        reduction_function = videomae_merge
    elif mode in ['drop', 'random_drop']:
        reduction_function = videomae_drop
    elif mode in ['hybrid']:
        reduction_function = videomae_hybrid

    for module in model.modules():
        if isinstance(module, Block) and not isinstance(module, ToMeDuplicateBlock):
            module.__class__ = ToMeBlock
            module._tome_info = model_wrapper._tome_info
            module.reduction_function = reduction_function
        elif isinstance(module, ToMeDuplicateBlock):
            module._tome_info = model_wrapper._tome_info
            module.reduction_function = reduction_function
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention