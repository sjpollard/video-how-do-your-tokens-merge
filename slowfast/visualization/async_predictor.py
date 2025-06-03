#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import atexit
import numpy as np
import queue
import torch
import torch.multiprocessing as mp

import slowfast.utils.logging as logging
from slowfast.datasets import cv2_transform

logger = logging.get_logger(__name__)


class AsyncVis:
    class _VisWorker(mp.Process):
        def __init__(self, video_vis, task_queue, result_queue):
            """
            Visualization Worker for AsyncVis.
            Args:
                video_vis (VideoVisualizer object): object with tools for visualization.
                task_queue (mp.Queue): a shared queue for incoming task for visualization.
                result_queue (mp.Queue): a shared queue for visualized results.
            """
            self.video_vis = video_vis
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            """
            Run visualization asynchronously.
            """
            while True:
                task = self.task_queue.get()
                if isinstance(task, _StopToken):
                    break

                frames = draw_predictions(task, self.video_vis)
                task.frames = np.array(frames)
                self.result_queue.put(task)

    def __init__(self, video_vis, n_workers=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            n_workers (Optional[int]): number of CPUs for running video visualizer.
                If not given, use all CPUs.
        """

        num_workers = mp.cpu_count() if n_workers is None else n_workers

        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.get_indices_ls = []
        self.procs = []
        self.result_data = {}
        self.put_id = -1
        for _ in range(max(num_workers, 1)):
            self.procs.append(
                AsyncVis._VisWorker(
                    video_vis, self.task_queue, self.result_queue
                )
            )

        for p in self.procs:
            p.start()

        atexit.register(self.shutdown)

    def put(self, task):
        """
        Add the new task to task queue.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames, boxes, predictions)
        """
        self.put_id += 1
        self.task_queue.put(task)

    def get(self):
        """
        Return visualized frames/clips in the correct order based on task id if
        result(s) is available. Otherwise, raise queue.Empty exception.
        """
        get_idx = self.get_indices_ls[0]
        if self.result_data.get(get_idx) is not None:
            res = self.result_data[get_idx]
            del self.result_data[get_idx]
            del self.get_indices_ls[0]
            return res

        while True:
            res = self.result_queue.get(block=False)
            idx = res.id
            if idx == get_idx:
                del self.get_indices_ls[0]
                return res
            self.result_data[idx] = res

    def __call__(self, task):
        """
        How many results are ready to be returned.
        """
        self.put(task)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(_StopToken())

    @property
    def result_available(self):
        return self.result_queue.qsize() + len(self.result_data)

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5


class _StopToken:
    pass


def draw_predictions(task, video_vis):
    """
    Draw prediction for the given task.
    Args:
        task (TaskInfo object): task object that contain
            the necessary information for visualization. (e.g. frames, preds)
            All attributes must lie on CPU devices.
        video_vis (VideoVisualizer object): the video visualizer object.
    """
    boxes = task.bboxes
    frames = task.frames
    preds = task.action_preds
    if boxes is not None:
        img_width = task.img_width
        img_height = task.img_height
        if boxes.device != torch.device("cpu"):
            boxes = boxes.cpu()
        boxes = cv2_transform.revert_scaled_boxes(
            task.crop_size, boxes, img_height, img_width
        )

    keyframe_idx = len(frames) // 2 - task.num_buffer_frames
    draw_range = [
        keyframe_idx - task.clip_vis_size,
        keyframe_idx + task.clip_vis_size,
    ]
    buffer = frames[: task.num_buffer_frames]
    frames = frames[task.num_buffer_frames :]
    if boxes is not None:
        if len(boxes) != 0:
            frames = video_vis.draw_clip_range(
                frames,
                preds,
                boxes,
                keyframe_idx=keyframe_idx,
                draw_range=draw_range,
            )
    else:
        frames = video_vis.draw_clip_range(
            frames, preds, keyframe_idx=keyframe_idx, draw_range=draw_range
        )
    del task

    return buffer + frames
