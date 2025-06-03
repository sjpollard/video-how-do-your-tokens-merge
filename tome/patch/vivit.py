# Based on: https://github.com/facebookresearch/ToMe/blob/main/tome/patch/timm.py

import copy
from typing import Optional, Union, Tuple

import torch
from torch import nn
from slowfast.models.vivit_video_model_builder import ViViT
from transformers.models.vivit.modeling_vivit import VivitLayer, VivitAttention, VivitSelfAttention

import math

from tome.merge import bipartite_soft_matching, bipartite_soft_matching_drop, bipartite_soft_matching_hybrid, merge_source, merge_wavg
from tome.utils import parse_r


class ToMeVivitLayer(VivitLayer):
    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        self_attention_outputs = self.attention(
            # in Vivit, layernorm is applied before self-attention
            self.layernorm_before(hidden_states),
            attn_size,
            self._tome_info["head_aggregation"],
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        metric = self_attention_outputs[1]
        # add self attentions if we output attention weights
        outputs = self_attention_outputs[2:]

        # first residual connection
        hidden_states = attention_output + hidden_states

        hidden_states = self.reduction_function(metric, hidden_states, self._tome_info)

        # in Vivit, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs
    

class ToMeDuplicateVivitLayer(ToMeVivitLayer):
    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        self_attention_outputs = self.attention(
            # in Vivit, layernorm is applied before self-attention
            self.layernorm_before(hidden_states),
            attn_size,
            self._tome_info["head_aggregation"],
            head_mask,
            output_attentions=output_attentions,
        )

        metric = self_attention_outputs[1]

        hidden_states = self.reduction_function(metric, hidden_states, self._tome_info)

        return [hidden_states]


class ToMeVivitAttention(VivitAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        size: torch.Tensor = None,
        head_aggregation: str = 'mean',
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, size, head_aggregation, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output, self_outputs[1]) + self_outputs[2:]  # add attentions if we output them
        return outputs


class ToMeVivitSelfAttention(VivitSelfAttention):
    def forward(
        self, hidden_states, size: torch.Tensor = None, head_aggregation: str = 'mean', head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        ## Apply proportional attention
        if size is not None:
            attention_scores = attention_scores + size.log()[:, None, None, :, 0]

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        if head_aggregation == 'mean':
            metric = key_layer.mean(1)
        elif head_aggregation == 'concat':
            metric = torch.cat(key_layer.split(1, dim=1), dim=-1).squeeze(dim=1)
            
        outputs = (context_layer, metric, attention_probs) if output_attentions else (context_layer, metric)

        return outputs


def vivit_merge(metric, x, _tome_info):
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


def vivit_drop(metric, x, _tome_info):
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


def vivit_hybrid(metric, x, _tome_info):
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
        model.vivit.encoder.layer.insert(index=i, module=copy.deepcopy(model.vivit.encoder.layer[i]))
        model.vivit.encoder.layer[i].__class__ = ToMeDuplicateVivitLayer
    model.vivit.config.num_hidden_layers = model.vivit.config.num_hidden_layers + quantity


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.vivit.encoder.layer), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().forward(*args, **kwdargs)

    return ToMeVisionTransformer


def apply_patch(
    model_wrapper: ViViT, trace_source: bool = False, prop_attn: bool = True, mode: str = 'merge',
    head_aggregation: str = 'mean', threshold: float = 0.0, verbose: bool = False
):  
    model = model_wrapper.vivit
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
        "class_token": model.embeddings.cls_token is not None,
        "distill_token": False,
        "mode": mode,
        "head_aggregation": head_aggregation,
        "threshold": threshold
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    if mode in ['merge', 'random_merge']:
        reduction_function = vivit_merge
    elif mode in ['drop', 'random_drop']:
        reduction_function = vivit_drop
    elif mode in ['hybrid']:
        reduction_function = vivit_hybrid

    for module in model.modules():
        if isinstance(module, VivitLayer) and not isinstance(module, ToMeDuplicateVivitLayer):
            module.__class__ = ToMeVivitLayer
            module._tome_info = model_wrapper._tome_info
            module.reduction_function = reduction_function
        elif isinstance(module, ToMeDuplicateVivitLayer):
            module._tome_info = model_wrapper._tome_info
            module.reduction_function = reduction_function
        elif isinstance(module, VivitAttention):
            module.__class__ = ToMeVivitAttention
        elif isinstance(module, VivitSelfAttention):
            module.__class__ = ToMeVivitSelfAttention