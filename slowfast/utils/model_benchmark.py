#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Benchmark a video classification model."""

import numpy as np
import torch
import wandb
import tome
from tqdm import tqdm

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
from slowfast.models import build_model

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_benchmark(model, cfg):
    # Enable eval mode.
    model.eval()

    total_iterations = cfg.MODEL_BENCHMARK.ITERATIONS + cfg.MODEL_BENCHMARK.WARMUP_ITERATIONS

    dummy_size = (cfg.TEST.BATCH_SIZE // cfg.NUM_GPUS, cfg.DATA.INPUT_CHANNEL_NUM[0], cfg.DATA.NUM_FRAMES, cfg.DATA.TEST_CROP_SIZE, cfg.DATA.TEST_CROP_SIZE)

    results = []

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for cur_iter in tqdm(range(total_iterations)):

        dummy_input = [torch.rand(dummy_size, device='cuda')]

        start.record()
        model(dummy_input)
        end.record()

        torch.cuda.synchronize()

        time = start.elapsed_time(end)
        results.append(sum(du.all_gather_unaligned(time)))

    average_frame_time = sum(results[cfg.MODEL_BENCHMARK.WARMUP_ITERATIONS:]) / (cfg.TEST.BATCH_SIZE * cfg.DATA.NUM_FRAMES * cfg.MODEL_BENCHMARK.ITERATIONS) / 1000.0
    average_fps = 1.0 / average_frame_time

    logger.info(f'Average time per frame is {average_frame_time}(s) after {cfg.MODEL_BENCHMARK.ITERATIONS} iterations')
    logger.info(f'Average fps is {average_fps}(im/s) after {cfg.MODEL_BENCHMARK.ITERATIONS} iterations')
    if cfg.WANDB.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):  
        wandb.log({'test/avg_frame_time': average_frame_time,
                   'test/avg_fps': average_fps})


def model_benchmark(cfg):
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Build the video model and print model statistics.
    model = build_model(cfg)

    cu.load_test_checkpoint(cfg, model)

    if cfg.TOME.ENABLE:
        if cfg.MODEL.MODEL_NAME == 'TimeSformer':
            patch_func = tome.patch.timesformer
            duplicate_func = tome.patch.duplicate_timesformer
        elif cfg.MODEL.MODEL_NAME == 'Motionformer':
            patch_func = tome.patch.motionformer
            duplicate_func = tome.patch.duplicate_motionformer
        elif cfg.MODEL.MODEL_NAME == 'ViViT':
            patch_func = tome.patch.vivit
            duplicate_func = tome.patch.duplicate_vivit
        elif cfg.MODEL.MODEL_NAME == 'VideoMAE':
            patch_func = tome.patch.videomae
            duplicate_func = tome.patch.duplicate_videomae
        if cfg.NUM_GPUS > 1:
            if cfg.TOME.LAYER_QUANTITY > 1:
                cfg.TOME.R_VALUE = [0] * cfg.TOME.LAYER_TO_DUPLICATE + [cfg.TOME.R_VALUE] * cfg.TOME.LAYER_QUANTITY + [0] * (11 - cfg.TOME.LAYER_TO_DUPLICATE)
                duplicate_func(model.module, layer_to_duplicate=cfg.TOME.LAYER_TO_DUPLICATE, quantity=cfg.TOME.LAYER_QUANTITY)
            patch_func(model.module, prop_attn=cfg.TOME.PROP_ATTN, mode=cfg.TOME.MODE, head_aggregation=cfg.TOME.HEAD_AGGREGATION)
            model.module.r = (cfg.TOME.R_VALUE, cfg.TOME.SCHEDULE)
        else:
            if cfg.TOME.LAYER_QUANTITY > 1:
                cfg.TOME.R_VALUE = [0] * cfg.TOME.LAYER_TO_DUPLICATE + [cfg.TOME.R_VALUE] * cfg.TOME.LAYER_QUANTITY + [0] * (11 - cfg.TOME.LAYER_TO_DUPLICATE)
                duplicate_func(model, layer_to_duplicate=cfg.TOME.LAYER_TO_DUPLICATE, quantity=cfg.TOME.LAYER_QUANTITY)
            patch_func(model, prop_attn=cfg.TOME.PROP_ATTN, mode=cfg.TOME.MODE, head_aggregation=cfg.TOME.HEAD_AGGREGATION)
            model.r = (cfg.TOME.R_VALUE, cfg.TOME.SCHEDULE)

    if cfg.WANDB.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):  
        wandb.init(project=f'{cfg.WANDB.PROJECT}', config=cfg)

    # # Perform multi-view test on the entire dataset.
    perform_benchmark(model, cfg)

    if cfg.WANDB.ENABLE and du.is_master_proc(
            cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        wandb.finish()