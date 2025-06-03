#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .build import MODEL_REGISTRY, build_model  # noqa
from .contrastive import ContrastiveModel  # noqa
from .custom_video_model_builder import *  # noqa
from .masked import MaskMViT  # noqa
from .video_model_builder import MViT, ResNet, SlowFast  # noqa
from .timesformer import TimeSformer # noqa
from .motionformer_video_model_builder import Motionformer # noqa
from .vivit_video_model_builder import ViViT # noqa
from .videomae_video_model_builder import VideoMAE # noqa

try:
    from .ptv_model_builder import (
        PTVCSN,
        PTVX3D,
        PTVR2plus1D,
        PTVResNet,
        PTVSlowFast,
    )  # noqa
except Exception:
    print("Please update your PyTorchVideo to latest master")
