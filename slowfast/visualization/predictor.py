#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import queue
import cv2
import torch

import slowfast.utils.checkpoint as cu
from slowfast.datasets import cv2_transform
from slowfast.models import build_model
from slowfast.utils import logging
from slowfast.visualization.utils import process_cv2_inputs

logger = logging.get_logger(__name__)

