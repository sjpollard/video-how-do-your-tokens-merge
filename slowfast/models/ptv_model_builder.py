#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


"""Video models using PyTorchVideo model builder."""

from functools import partial
import torch.nn as nn

from slowfast.models.batchnorm_helper import get_norm
from slowfast.models.video_model_builder import _POOL1, _TEMPORAL_KERNEL_BASIS

from pytorchvideo.models.csn import create_csn
from pytorchvideo.models.head import (
    create_res_basic_head,
    create_res_roi_pooling_head,
)
from pytorchvideo.models.r2plus1d import (
    create_2plus1d_bottleneck_block,
    create_r2plus1d,
)
from pytorchvideo.models.resnet import create_bottleneck_block, create_resnet
from pytorchvideo.models.slowfast import create_slowfast
from pytorchvideo.models.vision_transformers import (
    create_multiscale_vision_transformers,
)
from pytorchvideo.models.x3d import (
    Swish,
    create_x3d,
    create_x3d_bottleneck_block,
)

from .build import MODEL_REGISTRY


def get_head_act(act_func):
    """
    Return the actual head activation function given the activation fucntion name.

    Args:
        act_func (string): activation function to use. 'softmax': applies
        softmax on the output. 'sigmoid': applies sigmoid on the output.
    Returns:
        nn.Module: the activation layer.
    """
    if act_func == "softmax":
        return nn.Softmax(dim=1)
    elif act_func == "sigmoid":
        return nn.Sigmoid()
    else:
        raise NotImplementedError(
            "{} is not supported as a head activation "
            "function.".format(act_func)
        )


@MODEL_REGISTRY.register()
class PTVX3D(nn.Module):
    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(PTVX3D, self).__init__()

        assert (
            cfg.RESNET.STRIDE_1X1 is False
        ), "STRIDE_1x1 must be True for PTVX3D"
        assert (
            cfg.RESNET.TRANS_FUNC == "x3d_transform"
        ), f"Unsupported TRANS_FUNC type {cfg.RESNET.TRANS_FUNC} for PTVX3D"
        assert (
            cfg.DETECTION.ENABLE is False
        ), "Detection model is not supported for PTVX3D yet."

        self._construct_network(cfg)

    def _construct_network(self, cfg):
        """
        Builds a X3D model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """

        # Params from configs.
        norm_module = get_norm(cfg)
        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.model = create_x3d(
            # Input clip configs.
            input_channel=cfg.DATA.INPUT_CHANNEL_NUM[0],
            input_clip_length=cfg.DATA.NUM_FRAMES,
            input_crop_size=cfg.DATA.TRAIN_CROP_SIZE,
            # Model configs.
            model_num_class=cfg.MODEL.NUM_CLASSES,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            width_factor=cfg.X3D.WIDTH_FACTOR,
            depth_factor=cfg.X3D.DEPTH_FACTOR,
            # Normalization configs.
            norm=norm_module,
            norm_eps=1e-5,
            norm_momentum=0.1,
            # Activation configs.
            activation=partial(nn.ReLU, inplace=cfg.RESNET.INPLACE_RELU),
            # Stem configs.
            stem_dim_in=cfg.X3D.DIM_C1,
            stem_conv_kernel_size=(temp_kernel[0][0][0], 3, 3),
            stem_conv_stride=(1, 2, 2),
            # Stage configs.
            stage_conv_kernel_size=(
                (temp_kernel[1][0][0], 3, 3),
                (temp_kernel[2][0][0], 3, 3),
                (temp_kernel[3][0][0], 3, 3),
                (temp_kernel[4][0][0], 3, 3),
            ),
            stage_spatial_stride=(2, 2, 2, 2),
            stage_temporal_stride=(1, 1, 1, 1),
            bottleneck=create_x3d_bottleneck_block,
            bottleneck_factor=cfg.X3D.BOTTLENECK_FACTOR,
            se_ratio=0.0625,
            inner_act=Swish,
            # Head configs.
            head_dim_out=cfg.X3D.DIM_C5,
            head_pool_act=partial(nn.ReLU, inplace=cfg.RESNET.INPLACE_RELU),
            head_bn_lin5_on=cfg.X3D.BN_LIN5,
            head_activation=None,
            head_output_with_global_average=False,
        )

        self.post_act = get_head_act(cfg.MODEL.HEAD_ACT)

    def forward(self, x, bboxes=None):
        x = x[0]
        x = self.model(x)
        # Performs fully convlutional inference.
        if not self.training:
            x = self.post_act(x)
            x = x.mean([2, 3, 4])

        x = x.reshape(x.shape[0], -1)
        return x


@MODEL_REGISTRY.register()
class PTVCSN(nn.Module):
    """
    CSN models using PyTorchVideo model builder.
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(PTVCSN, self).__init__()

        assert (
            cfg.DETECTION.ENABLE is False
        ), "Detection model is not supported for PTVCSN yet."

        self._construct_network(cfg)

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """

        # Params from configs.
        norm_module = get_norm(cfg)

        self.model = create_csn(
            # Input clip configs.
            input_channel=cfg.DATA.INPUT_CHANNEL_NUM[0],
            # Model configs.
            model_depth=cfg.RESNET.DEPTH,
            model_num_class=cfg.MODEL.NUM_CLASSES,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            # Normalization configs.
            norm=norm_module,
            # Activation configs.
            activation=partial(nn.ReLU, inplace=cfg.RESNET.INPLACE_RELU),
            # Stem configs.
            stem_dim_out=cfg.RESNET.WIDTH_PER_GROUP,
            stem_conv_kernel_size=(3, 7, 7),
            stem_conv_stride=(1, 2, 2),
            stem_pool=nn.MaxPool3d,
            stem_pool_kernel_size=(1, 3, 3),
            stem_pool_stride=(1, 2, 2),
            # Stage configs.
            stage_conv_a_kernel_size=(1, 1, 1),
            stage_conv_b_kernel_size=(3, 3, 3),
            stage_conv_b_width_per_group=1,
            stage_spatial_stride=(1, 2, 2, 2),
            stage_temporal_stride=(1, 2, 2, 2),
            bottleneck=create_bottleneck_block,
            # Head configs.
            head_pool=nn.AvgPool3d,
            head_pool_kernel_size=(
                cfg.DATA.NUM_FRAMES // 8,
                cfg.DATA.TRAIN_CROP_SIZE // 32,
                cfg.DATA.TRAIN_CROP_SIZE // 32,
            ),
            head_activation=None,
            head_output_with_global_average=False,
        )

        self.post_act = get_head_act(cfg.MODEL.HEAD_ACT)

    def forward(self, x, bboxes=None):
        x = x[0]
        x = self.model(x)
        # Performs fully convlutional inference.
        if not self.training:
            x = self.post_act(x)
            x = x.mean([2, 3, 4])

        x = x.reshape(x.shape[0], -1)
        return x


@MODEL_REGISTRY.register()
class PTVR2plus1D(nn.Module):
    """
    R(2+1)D models using PyTorchVideo model builder.
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(PTVR2plus1D, self).__init__()

        assert (
            cfg.DETECTION.ENABLE is False
        ), "Detection model is not supported for PTVR2plus1D yet."

        self._construct_network(cfg)

    def _construct_network(self, cfg):
        """
        Builds a single pathway R(2+1)D model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        self.model = create_r2plus1d(
            # Input clip configs.
            input_channel=cfg.DATA.INPUT_CHANNEL_NUM[0],
            # Model configs.
            model_depth=cfg.RESNET.DEPTH,
            model_num_class=cfg.MODEL.NUM_CLASSES,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            # Normalization configs.
            norm=get_norm(cfg),
            norm_eps=1e-5,
            norm_momentum=0.1,
            # Activation configs.
            activation=partial(nn.ReLU, inplace=cfg.RESNET.INPLACE_RELU),
            # Stem configs.
            stem_dim_out=cfg.RESNET.WIDTH_PER_GROUP,
            stem_conv_kernel_size=(1, 7, 7),
            stem_conv_stride=(1, 2, 2),
            # Stage configs.
            stage_conv_a_kernel_size=(
                (1, 1, 1),
                (1, 1, 1),
                (1, 1, 1),
                (1, 1, 1),
            ),
            stage_conv_b_kernel_size=(
                (3, 3, 3),
                (3, 3, 3),
                (3, 3, 3),
                (3, 3, 3),
            ),
            stage_conv_b_num_groups=(1, 1, 1, 1),
            stage_conv_b_dilation=(
                (1, 1, 1),
                (1, 1, 1),
                (1, 1, 1),
                (1, 1, 1),
            ),
            stage_spatial_stride=(2, 2, 2, 2),
            stage_temporal_stride=(1, 1, 2, 2),
            stage_bottleneck=(
                create_2plus1d_bottleneck_block,
                create_2plus1d_bottleneck_block,
                create_2plus1d_bottleneck_block,
                create_2plus1d_bottleneck_block,
            ),
            # Head configs.
            head_pool=nn.AvgPool3d,
            head_pool_kernel_size=(
                cfg.DATA.NUM_FRAMES // 4,
                cfg.DATA.TRAIN_CROP_SIZE // 32,
                cfg.DATA.TRAIN_CROP_SIZE // 32,
            ),
            head_activation=None,
            head_output_with_global_average=False,
        )

        self.post_act = get_head_act(cfg.MODEL.HEAD_ACT)

    def forward(self, x, bboxes=None):
        x = x[0]
        x = self.model(x)
        # Performs fully convlutional inference.
        if not self.training:
            x = self.post_act(x)
            x = x.mean([2, 3, 4])

        x = x.view(x.shape[0], -1)
        return x


@MODEL_REGISTRY.register()
class PTVMViT(nn.Module):
    """
    MViT models using PyTorchVideo model builder.
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(PTVMViT, self).__init__()

        assert (
            cfg.DETECTION.ENABLE is False
        ), "Detection model is not supported for PTVMViT yet."

        self._construct_network(cfg)

    def _construct_network(self, cfg):
        """
        Builds a MViT model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        self.model = create_multiscale_vision_transformers(
            spatial_size=cfg.DATA.TRAIN_CROP_SIZE,
            temporal_size=cfg.DATA.NUM_FRAMES,
            cls_embed_on=cfg.MVIT.CLS_EMBED_ON,
            sep_pos_embed=cfg.MVIT.SEP_POS_EMBED,
            depth=cfg.MVIT.DEPTH,
            norm=cfg.MVIT.NORM,
            # Patch embed config.
            input_channels=cfg.DATA.INPUT_CHANNEL_NUM[0],
            patch_embed_dim=cfg.MVIT.EMBED_DIM,
            conv_patch_embed_kernel=cfg.MVIT.PATCH_KERNEL,
            conv_patch_embed_stride=cfg.MVIT.PATCH_STRIDE,
            conv_patch_embed_padding=cfg.MVIT.PATCH_PADDING,
            enable_patch_embed_norm=cfg.MVIT.NORM_STEM,
            use_2d_patch=cfg.MVIT.PATCH_2D,
            # Attention block config.
            num_heads=cfg.MVIT.NUM_HEADS,
            mlp_ratio=cfg.MVIT.MLP_RATIO,
            qkv_bias=cfg.MVIT.QKV_BIAS,
            dropout_rate_block=cfg.MVIT.DROPOUT_RATE,
            droppath_rate_block=cfg.MVIT.DROPPATH_RATE,
            pooling_mode=cfg.MVIT.MODE,
            pool_first=cfg.MVIT.POOL_FIRST,
            embed_dim_mul=cfg.MVIT.DIM_MUL,
            atten_head_mul=cfg.MVIT.HEAD_MUL,
            pool_q_stride_size=cfg.MVIT.POOL_Q_STRIDE,
            pool_kv_stride_size=cfg.MVIT.POOL_KV_STRIDE,
            pool_kv_stride_adaptive=cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE,
            pool_kvq_kernel=cfg.MVIT.POOL_KVQ_KERNEL,
            # Head config.
            head_dropout_rate=cfg.MODEL.DROPOUT_RATE,
            head_num_classes=cfg.MODEL.NUM_CLASSES,
        )

        self.post_act = get_head_act(cfg.MODEL.HEAD_ACT)

    def forward(self, x, bboxes=None):
        x = x[0]
        x = self.model(x)

        if not self.training:
            x = self.post_act(x)

        return x
