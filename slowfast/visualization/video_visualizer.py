#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import itertools
import logging as log
import numpy as np
import matplotlib.pyplot as plt
import torch

import slowfast.utils.logging as logging
from slowfast.utils.misc import get_class_names

logger = logging.get_logger(__name__)
log.getLogger("matplotlib").setLevel(log.ERROR)


def _create_text_labels(classes, scores, class_names, ground_truth=False):
    """
    Create text labels.
    Args:
        classes (list[int]): a list of class ids for each example.
        scores (list[float] or None): list of scores for each example.
        class_names (list[str]): a list of class names, ordered by their ids.
        ground_truth (bool): whether the labels are ground truth.
    Returns:
        labels (list[str]): formatted text labels.
    """
    try:
        labels = [class_names[i] for i in classes]
    except IndexError:
        logger.error("Class indices get out of range: {}".format(classes))
        return None

    if ground_truth:
        labels = ["[{}] {}".format("GT", label) for label in labels]
    elif scores is not None:
        assert len(classes) == len(scores)
        labels = [
            "[{:.2f}] {}".format(s, label) for s, label in zip(scores, labels)
        ]
    return labels