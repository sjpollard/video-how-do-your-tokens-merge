# Based on: https://github.com/facebookresearch/ToMe/blob/main/tome/patch/timm.py

import torch

from slowfast.models.timesformer import TimeSformer, Block, Attention
from einops import rearrange

from tome.merge import bipartite_soft_matching, bipartite_soft_matching_drop, bipartite_soft_matching_hybrid, merge_source, merge_wavg
from tome.utils import parse_r


class ToMeBlock(Block):
    def forward(self, x, B, T, W):
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            ## Temporal
            xt = x[:,1:,:]
            xt = rearrange(xt, 'b (p t) m -> (b p) t m',b=B,p=num_spatial_tokens,t=T)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b p) t m -> b (p t) m',b=B,p=num_spatial_tokens,t=T)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:,1:,:] + res_temporal

            ## Spatial
            init_cls_token = x[:,0,:].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
            xs = xt
            xs = rearrange(xs, 'b (p t) m -> (b t) (p) m',b=B,p=num_spatial_tokens,t=T)
            xs = torch.cat((cls_token, xs), 1)
            res_spatial, metric = self.attn(self.norm1(xs), attn_size)
            res_spatial = self.drop_path(res_spatial)

            ### Taking care of CLS token
            cls_token = res_spatial[:,0,:]
            cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=T)
            cls_token = torch.mean(cls_token,1,True) ## averaging for every frame
            res_spatial = res_spatial[:,1:,:]
            num_spatial_tokens = res_spatial.size(1)
            res_spatial = rearrange(res_spatial, '(b t) p m -> b (p t) m',b=B,p=num_spatial_tokens,t=T)
            res = res_spatial
            x = xt

            ## Mlp
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)

            x = self.reduction_function(metric, x, self._tome_info, B, T, num_spatial_tokens)

            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class ToMeAttention(Attention):
    def forward(self, x, size: torch.Tensor = None):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale

        ## Apply proportional attention
        if size is not None:
            attn[:, :, 1:, 1:] += size.log()[:, None, None, :, 0]
            
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x, k.mean(1)[:, 1:, :]

def timesformer_merge(metric, x, _tome_info, B, T, num_spatial_tokens):
    #Place ToMe here
    r = _tome_info["r"].pop(0)
    if r > 0:
        cls, merged_x = x[:, 0:1, :], x[:, 1:, :]
        merged_x = rearrange(merged_x, f'b (p t) m -> (b t) p m', b=B, t=T, p=num_spatial_tokens)
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
                merge, merged_x, _tome_info["source"]
            )
        pre_merge = merged_x.size(1)
        merged_x, _tome_info["size"] = merge_wavg(merge, merged_x, _tome_info["size"])
        if _tome_info['verbose']:
            print(f'Merged {pre_merge} to {merged_x.size(1)} tokens')
        x = torch.cat((cls, rearrange(merged_x, f'(b t) p m -> b (p t) m', b=B, t=T)), dim=1)
        
    return x


def timesformer_drop(metric, x, _tome_info, B, T, num_spatial_tokens):
    r = _tome_info["r"].pop(0)
    if r > 0:
        cls, dropped_x = x[:, 0:1, :], x[:, 1:, :]
        dropped_x = rearrange(dropped_x, f'b (p t) m -> (b t) p m', b=B, t=T, p=num_spatial_tokens)
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
                n, t, _ = dropped_x.shape
                _tome_info["source"] = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)
            _tome_info["source"] = drop(
                _tome_info["source"]
            )
        pre_drop = dropped_x.size(1)
        dropped_x = drop(dropped_x)
        _tome_info["size"] = torch.ones((dropped_x.size(0), dropped_x.size(1), 1), device=x.device)
        if _tome_info['verbose']:
            print(f'Dropped {pre_drop} to {dropped_x.size(1)} tokens')
        x = torch.cat((cls, rearrange(dropped_x, f'(b t) p m -> b (p t) m', b=B, t=T)), dim=1)
        
    return x


def timesformer_hybrid(metric, x, _tome_info, B, T, num_spatial_tokens):
    #Place ToMe here
    r = _tome_info["r"].pop(0)
    if r > 0:
        cls, merged_x = x[:, 0:1, :], x[:, 1:, :]
        merged_x = rearrange(merged_x, f'b (p t) m -> (b t) p m', b=B, t=T, p=num_spatial_tokens)
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
                merge, merged_x, _tome_info["source"]
            )
        pre_merge = merged_x.size(1)
        merged_x, _tome_info["size"] = merge_wavg(merge, merged_x, _tome_info["size"])
        if _tome_info['verbose']:
            print(f'Merged {pre_merge} to {merged_x.size(1)} tokens')
        x = torch.cat((cls, rearrange(merged_x, f'(b t) p m -> b (p t) m', b=B, t=T)), dim=1)
        
    return x


def apply_duplicate_patch(model, layer_to_duplicate, quantity):
    for i in range(layer_to_duplicate + 1, layer_to_duplicate + quantity):
        model.model.blocks.insert(index=i, module=model.model.blocks[layer_to_duplicate])


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.model.blocks), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().forward(*args, **kwdargs)

    return ToMeVisionTransformer


def apply_patch(
    model_wrapper: TimeSformer, trace_source: bool = False, prop_attn: bool = True, mode: str = 'merge',
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
        "threshold": threshold
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    if mode in ['merge', 'random_merge']:
        reduction_function = timesformer_merge
    elif mode in ['drop', 'random_drop']:
        reduction_function = timesformer_drop
    elif mode in ['hybrid']:
        reduction_function = timesformer_hybrid

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = ToMeBlock
            module._tome_info = model_wrapper._tome_info
            module.reduction_function = reduction_function
            module.attn.__class__ = ToMeAttention
