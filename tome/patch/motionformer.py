# Based on: https://github.com/facebookresearch/ToMe/blob/main/tome/patch/timm.py

import torch

from slowfast.models.motionformer_video_model_builder import Motionformer
from slowfast.models import motionformer_nystrom_helper as nystrom_helper, motionformer_orthoformer_helper as orthoformer_helper, motionformer_performer_helper as performer_helper
from slowfast.models.motionformer_vit_helper import Block, TrajectoryAttention, qkv_attn
from einops import rearrange

from tome.merge import bipartite_soft_matching, bipartite_soft_matching_drop, bipartite_soft_matching_hybrid, merge_source, merge_wavg
from tome.utils import parse_r


class ToMeBlock(Block):
    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        attn_out, _, metric = self.attn(
                self.norm1(x), 
                seq_len=seq_len, 
                num_frames=num_frames, 
                approx=approx,
                num_landmarks=num_landmarks,
                size=attn_size
        )
        x = x + self.drop_path(attn_out)

        x = self.reduction_function(metric, x, self._tome_info, num_frames)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ToMeTrajectoryAttention(TrajectoryAttention):
    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128, size: torch.Tensor = None):
        B, N, C = x.shape
        P = seq_len
        new_seq_len = (N - 1) // num_frames
        P = new_seq_len
        F = num_frames
        h = self.num_heads

        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # remove CLS token from q, k, v
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
            lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

        # let CLS token attend to key / values of all patches across time and space
        cls_out = qkv_attn(cls_q * self.scale, k, v)
        cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h)

        if approx == "nystrom":
            ## Shared spatial landmarks
            q_, k_, v_ = map(
                lambda t: rearrange(t, f'b h p d -> (b h) p d', h=h), (q_, k_, v_))
            x = nystrom_helper.nystrom_spatial_attn(
                q_, k_, v_,
                landmarks=num_landmarks,
                num_frames=F,
                inv_iters=6,
                use_spatial_landmarks=True
            )
            x = rearrange(x, f'(b h) p f d -> b h p f d', f=F, h=h)
        elif approx == "orthoformer":
            x = orthoformer_helper.orthoformer(
                q_, k_, v_,
                num_landmarks=num_landmarks,
                num_frames=F,
            )
        elif approx == "performer":
            # Form random projection matrices:
            m = 256 # r = 2m, m <= d
            d = self.head_dim
            seed = torch.ceil(torch.abs(torch.sum(q_) * performer_helper.BIG_CONSTANT))
            seed = torch.tensor(seed)
            projection_matrix = performer_helper.create_projection_matrix(
                m, d, seed=seed, device=q_.device, dtype=q_.dtype)
            q_, k_ = map(lambda t: rearrange(t, f'b h p d -> b p h d'), (q_, k_))
            q_prime = performer_helper.softmax_kernel_transformation(
                q_, 
                is_query=True, 
                projection_matrix=projection_matrix
            )
            k_prime = performer_helper.softmax_kernel_transformation(
                k_, 
                is_query=False, 
                projection_matrix=projection_matrix
            )
            q_prime, k_prime = map(
                lambda t: rearrange(t, f'b p h r -> b h p r'), (q_prime, k_prime))
            k_prime = rearrange(k_prime, 'b h (f n) r -> b h f n r', f=F)
            v_ = rearrange(v_, 'b h (f n) d -> b h f n d', f=F)
            kv = torch.einsum('b h f n r, b h f n d -> b h f r d', k_prime, v_)
            qkv = torch.einsum('b h p r, b h f r d -> b h p f d', q_prime, kv)
            normaliser = torch.einsum('b h f n r -> b h f r', k_prime)
            normaliser = torch.einsum('b h p r, b h f r -> b h p f', q_prime, normaliser)
            x = qkv / normaliser.unsqueeze(-1)
        else:
            # Using full attention
            q_dot_k = q_ @ k_.transpose(-2, -1)
            q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F) * self.scale
            if size is not None:
                q_dot_k = rearrange(q_dot_k, '(b h) q f n -> b h q (f n)', h=h, f=F)
                size = rearrange(size, f'(b f) s i -> b (s f) i', f=F)
                q_dot_k = q_dot_k + size.log()[:, None, None, :, 0]
                q_dot_k = rearrange(q_dot_k, 'b h q (f n) -> (b h) q f n', h=h, f=F)
            space_attn = (q_dot_k).softmax(dim=-1)
            attn = self.attn_drop(space_attn)
            v_ = rearrange(v_, 'b (f n) d -> b f n d', f=F, n=P)
            x = torch.einsum('b q f n, b f n d -> b q f d', attn, v_)

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B)
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F)
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2)
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F)
        q2 = self.proj_q(x_diag)
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1)
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)
        attn = attn.softmax(dim=-1)
        if self.use_original_code:
            x = rearrange(x, f'b s f (h d) -> b h s f d', f=F,  h=h)
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x)
        else:
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)')
        
        # concat back the cls token
        x = torch.cat((cls_out, x), dim=1)

        x = self.proj(x)
        x = self.proj_drop(x)

        k = rearrange(k_, f'(b h) (s f) d -> (b f) h s d', f=F, h=h)
        return x, attn, k.mean(1)


def motionformer_merge(metric, x, _tome_info, num_frames):
    r = _tome_info["r"].pop(0)
    if r > 0:
        cls, merged_x = x[:, 0:1, :], x[:, 1:, :]
        merged_x = rearrange(merged_x, f'b (s f) d -> (b f) s d', f=num_frames)
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
        x = torch.cat((cls, rearrange(merged_x, f'(b f) s d -> b (s f) d', f=num_frames)), dim=1)

    return x


def motionformer_drop(metric, x, _tome_info, num_frames):
    r = _tome_info["r"].pop(0)
    if r > 0:
        cls, dropped_x = x[:, 0:1, :], x[:, 1:, :]
        dropped_x = rearrange(dropped_x, f'b (s f) d -> (b f) s d', f=num_frames)
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
        x = torch.cat((cls, rearrange(dropped_x, f'(b f) s d -> b (s f) d', f=num_frames)), dim=1)
        
    return x


def motionformer_hybrid(metric, x, _tome_info, num_frames):
    r = _tome_info["r"].pop(0)
    if r > 0:
        cls, merged_x = x[:, 0:1, :], x[:, 1:, :]
        merged_x = rearrange(merged_x, f'b (s f) d -> (b f) s d', f=num_frames)
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
        x = torch.cat((cls, rearrange(merged_x, f'(b f) s d -> b (s f) d', f=num_frames)), dim=1)

    return x


def apply_duplicate_patch(model, layer_to_duplicate, quantity):
    for i in range(layer_to_duplicate + 1, layer_to_duplicate + quantity):
        model.blocks.insert(index=i, module=model.blocks[layer_to_duplicate])


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.blocks), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().forward(*args, **kwdargs)

    return ToMeVisionTransformer


def apply_patch(
    model: Motionformer, trace_source: bool = False, prop_attn: bool = True, mode: str = 'merge',
    head_aggregation: str = 'mean', threshold: float = 0.0, verbose: bool = False
):  
    ToMeVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model._tome_info = {
        "r": model.r,
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
        reduction_function = motionformer_merge
    elif mode in ['drop', 'random_drop']:
        reduction_function = motionformer_drop
    elif mode in ['hybrid']:
        reduction_function = motionformer_hybrid

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
            module.reduction_function = reduction_function
        elif isinstance(module, TrajectoryAttention):
            module.__class__ = ToMeTrajectoryAttention
