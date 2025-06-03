#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import os
import cv2
import torch
import tqdm

import slowfast.utils.checkpoint as cu
import slowfast.utils.logging as logging
from slowfast.datasets.ava_helper import parse_bboxes_file
from slowfast.datasets.cv2_transform import scale, scale_boxes
from slowfast.datasets.utils import get_sequence
from slowfast.models import build_model
from slowfast.utils import misc
from slowfast.utils.env import pathmgr
from slowfast.visualization.utils import process_cv2_inputs

logger = logging.get_logger(__name__)


def merge_pred_gt_boxes(pred_dict, gt_dict=None):
    """
    Merge data from precomputed and ground-truth boxes dictionaries.
    Args:
        pred_dict (dict): a dict which maps from `frame_idx` to a list of `boxes`
            and `labels`. Each `box` is a list of 4 box coordinates. `labels[i]` is
            a list of labels for `boxes[i]`.
        gt_dict (Optional[dict]): a dict which maps from `frame_idx` to a list of `boxes`
            and `labels`. Each `box` is a list of 4 box coordinates. `labels[i]` is
            a list of labels for `boxes[i]`. Note that label is -1 for predicted boxes.
    Returns:
        merged_dict (dict): merged dictionary from `pred_dict` and `gt_dict` if given.
            It is a dict which maps from `frame_idx` to a list of [`is_gt`, `boxes`, `labels`],
            where `is_gt` is a boolean indicate whether the `boxes` and `labels` are ground-truth.
    """
    merged_dict = {}
    for key, item in pred_dict.items():
        merged_dict[key] = [[False, item[0], item[1]]]

    if gt_dict is not None:
        for key, item in gt_dict.items():
            if merged_dict.get(key) is None:
                merged_dict[key] = [[True, item[0], item[1]]]
            else:
                merged_dict[key].append([True, item[0], item[1]])
    return merged_dict


def load_boxes_labels(cfg, video_name, fps, img_width, img_height):
    """
    Loading boxes and labels from AVA bounding boxes csv files.
    Args:
        cfg (CfgNode): config.
        video_name (str): name of the given video.
        fps (int or float): frames per second of the input video/images folder.
        img_width (int): width of images in input video/images folder.
        img_height (int): height of images in input video/images folder.
    Returns:
        preds_boxes (dict): a dict which maps from `frame_idx` to a list of `boxes`
            and `labels`. Each `box` is a list of 4 box coordinates. `labels[i]` is
            a list of labels for `boxes[i]`. Note that label is -1 for predicted boxes.
        gt_boxes (dict): if cfg.DEMO.GT_BOXES is given, return similar dict as
            all_pred_boxes but for ground-truth boxes.
    """
    starting_second = cfg.DEMO.STARTING_SECOND

    def sec_to_frameidx(sec):
        return (sec - starting_second) * fps

    def process_bboxes_dict(dictionary):
        """
        Replace all `keyframe_sec` in `dictionary` with `keyframe_idx` and
        merge all [`box_coordinate`, `box_labels`] pairs into
        [`all_boxes_coordinates`, `all_boxes_labels`] for each `keyframe_idx`.
        Args:
            dictionary (dict): a dictionary which maps `frame_sec` to a list of `box`.
                Each `box` is a [`box_coord`, `box_labels`] where `box_coord` is the
                coordinates of box and 'box_labels` are the corresponding
                labels for the box.
        Returns:
            new_dict (dict): a dict which maps from `frame_idx` to a list of `boxes`
                and `labels`. Each `box` in `boxes` is a list of 4 box coordinates. `labels[i]`
                is a list of labels for `boxes[i]`. Note that label is -1 for predicted boxes.
        """
        # Replace all keyframe_sec with keyframe_idx.
        new_dict = {}
        for keyframe_sec, boxes_and_labels in dictionary.items():
            # Ignore keyframes with no boxes
            if len(boxes_and_labels) == 0:
                continue
            keyframe_idx = sec_to_frameidx(keyframe_sec)
            boxes, labels = list(zip(*boxes_and_labels))
            # Shift labels from [1, n_classes] to [0, n_classes - 1].
            labels = [[i - 1 for i in box_label] for box_label in labels]
            boxes = np.array(boxes)
            boxes[:, [0, 2]] *= img_width
            boxes[:, [1, 3]] *= img_height
            new_dict[keyframe_idx] = [boxes.tolist(), list(labels)]
        return new_dict

    preds_boxes_path = cfg.DEMO.PREDS_BOXES
    gt_boxes_path = cfg.DEMO.GT_BOXES

    preds_boxes, _, _ = parse_bboxes_file(
        ann_filenames=[preds_boxes_path],
        ann_is_gt_box=[False],
        detect_thresh=cfg.AVA.DETECTION_SCORE_THRESH,
        boxes_sample_rate=1,
    )
    preds_boxes = preds_boxes[video_name]
    if gt_boxes_path == "":
        gt_boxes = None
    else:
        gt_boxes, _, _ = parse_bboxes_file(
            ann_filenames=[gt_boxes_path],
            ann_is_gt_box=[True],
            detect_thresh=cfg.AVA.DETECTION_SCORE_THRESH,
            boxes_sample_rate=1,
        )
        gt_boxes = gt_boxes[video_name]

    preds_boxes = process_bboxes_dict(preds_boxes)
    if gt_boxes is not None:
        gt_boxes = process_bboxes_dict(gt_boxes)

    return preds_boxes, gt_boxes
