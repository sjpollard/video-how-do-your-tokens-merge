#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified Model definition

"""Video models."""

from collections import OrderedDict
import copy
import math
import os
import random
import string
import torchvision
from torchvision.utils import make_grid, save_image
import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from . import motionformer_vit_helper as vit_helper
from .build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class Motionformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, cfg):
        super().__init__()
        self.img_size = cfg.DATA.TRAIN_CROP_SIZE
        self.patch_size = cfg.MOTIONFORMER.PATCH_SIZE
        self.in_chans = cfg.MOTIONFORMER.CHANNELS
        self.num_classes = cfg.MODEL.NUM_CLASSES if cfg.EPICKITCHENS.NUM_CLASSES is None else cfg.EPICKITCHENS.NUM_CLASSES
        self.embed_dim = cfg.MOTIONFORMER.EMBED_DIM
        self.depth = cfg.MOTIONFORMER.DEPTH
        self.num_heads = cfg.MOTIONFORMER.NUM_HEADS
        self.mlp_ratio = cfg.MOTIONFORMER.MLP_RATIO
        self.qkv_bias = cfg.MOTIONFORMER.QKV_BIAS
        self.drop_rate = cfg.MOTIONFORMER.DROP
        self.drop_path_rate = cfg.MOTIONFORMER.DROP_PATH
        self.head_dropout = cfg.MOTIONFORMER.HEAD_DROPOUT
        self.video_input = cfg.MOTIONFORMER.VIDEO_INPUT
        self.temporal_resolution = cfg.MOTIONFORMER.TEMPORAL_RESOLUTION
        self.use_mlp = cfg.MOTIONFORMER.USE_MLP
        self.num_features = self.embed_dim
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.attn_drop_rate = cfg.MOTIONFORMER.ATTN_DROPOUT
        self.head_act = cfg.MOTIONFORMER.HEAD_ACT
        self.cfg = cfg

        # Patch Embedding
        self.patch_embed = vit_helper.PatchEmbed(
            img_size=224, 
            patch_size=self.patch_size, 
            in_chans=self.in_chans, 
            embed_dim=self.embed_dim
        )

        # 3D Patch Embedding
        self.patch_embed_3d = vit_helper.PatchEmbed3D(
            img_size=self.img_size, 
            temporal_resolution=self.temporal_resolution, 
            patch_size=self.patch_size,
            in_chans=self.in_chans, 
            embed_dim=self.embed_dim, 
            z_block_size=self.cfg.MOTIONFORMER.PATCH_SIZE_TEMP
        )
        self.patch_embed_3d.proj.weight.data = torch.zeros_like(
            self.patch_embed_3d.proj.weight.data)
        
        # Number of patches
        if self.video_input:
            num_patches = self.patch_embed.num_patches * self.temporal_resolution
        else:
            num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.cls_token, std=.02)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=cfg.MOTIONFORMER.POS_DROPOUT)
        trunc_normal_(self.pos_embed, std=.02)

        if self.cfg.MOTIONFORMER.POS_EMBED == "joint":
            self.st_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, self.embed_dim))
            trunc_normal_(self.st_embed, std=.02)
        elif self.cfg.MOTIONFORMER.POS_EMBED == "separate":
            self.temp_embed = nn.Parameter(
                torch.zeros(1, self.temporal_resolution, self.embed_dim))

        # Layer Blocks
        dpr = [x.item() for x in torch.linspace(
            0, self.drop_path_rate, self.depth)]
        if self.cfg.MOTIONFORMER.ATTN_LAYER == "divided":
            self.blocks = nn.ModuleList([
                vit_helper.DividedSpaceTimeBlock(
                    attn_type=cfg.MOTIONFORMER.ATTN_LAYER, 
                    dim=self.embed_dim, 
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio, 
                    qkv_bias=self.qkv_bias, 
                    drop=self.drop_rate, 
                    attn_drop=self.attn_drop_rate, 
                    drop_path=dpr[i], 
                    norm_layer=norm_layer, 
                )
                for i in range(self.depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                vit_helper.Block(
                    attn_type=cfg.MOTIONFORMER.ATTN_LAYER, 
                    dim=self.embed_dim, 
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio, 
                    qkv_bias=self.qkv_bias, 
                    drop=self.drop_rate, 
                    attn_drop=self.attn_drop_rate, 
                    drop_path=dpr[i], 
                    norm_layer=norm_layer,
                    use_original_code=self.cfg.MOTIONFORMER.USE_ORIGINAL_TRAJ_ATTN_CODE
                )
                for i in range(self.depth)
            ])
        self.norm = norm_layer(self.embed_dim)

        # MLP head
        if self.use_mlp:
            hidden_dim = self.embed_dim
            if self.head_act == 'tanh':
                print("Using TanH activation in MLP")
                act = nn.Tanh() 
            elif self.head_act == 'gelu':
                print("Using GELU activation in MLP")
                act = nn.GELU()
            else:
                print("Using ReLU activation in MLP")
                act = nn.ReLU()
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(self.embed_dim, hidden_dim)),
                ('act', act),
            ]))
        else:
            self.pre_logits = nn.Identity()
        
        # Classifier Head
        self.head_drop = nn.Dropout(p=self.head_dropout)
        if isinstance(self.num_classes, (list)) and len(self.num_classes) > 1:
            for a, i in enumerate(range(len(self.num_classes))):
                setattr(self, "head%d"%a, nn.Linear(self.embed_dim, self.num_classes[i]))
        else:
            self.head = (nn.Linear(self.embed_dim, self.num_classes) 
                if self.num_classes > 0 else nn.Identity())

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.cfg.MOTIONFORMER.POS_EMBED == "joint":
            return {'pos_embed', 'cls_token', 'st_embed'}
        else:
            return {'pos_embed', 'cls_token', 'temp_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = (nn.Linear(self.embed_dim, num_classes) if num_classes > 0
            else nn.Identity())

    def forward_features(self, x):
        if self.video_input:
            x = x[0]
        B = x.shape[0]

        # Tokenize input
        if self.cfg.MOTIONFORMER.PATCH_SIZE_TEMP > 1:
            x = self.patch_embed_3d(x)
        else:
            # 2D tokenization
            if self.video_input:
                x = x.permute(0, 2, 1, 3, 4)
                (B, T, C, H, W) = x.shape
                x = x.reshape(B*T, C, H, W)

            x = self.patch_embed(x)

            if self.video_input:
                (B2, T2, D2) = x.shape
                x = x.reshape(B, T*T2, D2)

        # Append CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Interpolate positinoal embeddings
        if self.cfg.DATA.TRAIN_CROP_SIZE != 224:
            pos_embed = self.pos_embed
            N = pos_embed.shape[1] - 1
            npatch = int((x.size(1) - 1) / self.temporal_resolution)
            class_emb = pos_embed[:, 0]
            pos_embed = pos_embed[:, 1:]
            dim = x.shape[-1]
            pos_embed = torch.nn.functional.interpolate(
                pos_embed.reshape(
                    1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
                    0, 3, 1, 2),
                scale_factor=math.sqrt(npatch / N),
                mode='bicubic',
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            new_pos_embed = torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)
        else:
            new_pos_embed = self.pos_embed
            npatch = self.patch_embed.num_patches

        # Add positional embeddings to input
        if self.video_input:
            if self.cfg.MOTIONFORMER.POS_EMBED == "separate":
                cls_embed = self.pos_embed[:, 0, :].unsqueeze(1)
                tile_pos_embed = new_pos_embed[:, 1:, :].repeat(
                    1, self.temporal_resolution, 1)
                tile_temporal_embed = self.temp_embed.repeat_interleave(
                    npatch, 1)
                total_pos_embed = tile_pos_embed + tile_temporal_embed
                total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)
                x = x + total_pos_embed
            elif self.cfg.MOTIONFORMER.POS_EMBED == "joint":
                x = x + self.st_embed
        else:
            # image input
            x = x + new_pos_embed
                            
        # Apply positional dropout
        x = self.pos_drop(x)

        # Encoding using transformer layers
        for i, blk in enumerate(self.blocks):
            x = blk(
                x,
                seq_len=npatch,
                num_frames=self.temporal_resolution,
                approx=self.cfg.MOTIONFORMER.APPROX_ATTN_TYPE,
                num_landmarks=self.cfg.MOTIONFORMER.APPROX_ATTN_DIM
            )

        x = self.norm(x)[:, 0]
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head_drop(x)
        if isinstance(self.num_classes, (list,)) and len(self.num_classes) > 1:
            output = []
            for head in range(len(self.num_classes)):
                x_out = getattr(self, "head%d"%head)(x)
                if not self.training:
                    x_out = torch.nn.functional.softmax(x_out, dim=-1)
                output.append(x_out)
            return output
        else:
            x = self.head(x)
            if not self.training:
                x = torch.nn.functional.softmax(x, dim=-1)
            return x