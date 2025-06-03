#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
import wandb
import tome
import gc

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.env import pathmgr
from slowfast.utils.meters import AVAMeter, TestMeter, EPICTestMeter

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels, video_idx, time, meta) in enumerate(
        test_loader
    ):

        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            # Transfer the data to the current GPU device.
            if isinstance(labels, (dict,)):
                labels = {k: v.cuda() for k, v in labels.items()}
            else:
                labels = labels.cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        if not isinstance(val[i], (str,)):
                            val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = (
                ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            )
            metadata = (
                metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
            )

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds, ori_boxes, metadata)
            test_meter.log_iter_stats(None, cur_iter)
        elif cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel":
            if not cfg.CONTRASTIVE.KNN_ON:
                test_meter.finalize_metrics()
                return test_meter
            # preds = model(inputs, video_idx, time)
            train_labels = (
                model.module.train_labels
                if hasattr(model, "module")
                else model.train_labels
            )
            yd, yi = model(inputs, video_idx, time)
            batchSize = yi.shape[0]
            K = yi.shape[1]
            C = cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM  # eg 400 for Kinetics400
            candidates = train_labels.view(1, -1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)
            retrieval_one_hot = torch.zeros((batchSize * K, C)).cuda()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(cfg.CONTRASTIVE.T).exp_()
            probs = torch.mul(
                retrieval_one_hot.view(batchSize, -1, C),
                yd_transform.view(batchSize, -1, 1),
            )
            preds = torch.sum(probs, 1)
        else:
            # Perform the forward pass.
            preds = model(inputs)

        # Gather all the predictions across all the devices to perform ensemble.
        if isinstance(labels, (dict,)):
            if cfg.NUM_GPUS > 1:
                verb_preds, verb_labels, video_idx = du.all_gather(
                    [preds[0], labels['verb'], video_idx]
                )

                noun_preds, noun_labels, video_idx = du.all_gather(
                    [preds[1], labels['noun'], video_idx]
                )
                meta = du.all_gather_unaligned(meta)
                metadata = {'narration_id': []}
                for i in range(len(meta)):
                    metadata['narration_id'].extend(meta[i]['narration_id'])
            else:
                metadata = meta
                verb_preds, verb_labels, video_idx = preds[0], labels['verb'], video_idx
                noun_preds, noun_labels, video_idx = preds[1], labels['noun'], video_idx
            torch.cuda.synchronize()
            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(
                (verb_preds.detach().cpu(), noun_preds.detach().cpu()),
                (verb_labels.detach().cpu(), noun_labels.detach().cpu()),
                metadata,
                video_idx.detach().cpu()
            )
            test_meter.log_iter_stats(cur_iter)
        else:
            if cfg.NUM_GPUS > 1:
                preds, labels, video_idx = du.all_gather([preds, labels, video_idx])
            if cfg.NUM_GPUS:
                preds = preds.cpu()
                labels = labels.cpu()
                video_idx = video_idx.cpu()

            torch.cuda.synchronize()
            test_meter.iter_toc()

            if not cfg.VIS_MASK.ENABLE:
                # Update and log stats.
                test_meter.update_stats(
                    preds.detach(), labels.detach(), video_idx.detach()
                )
            test_meter.log_iter_stats(cur_iter)
        gc.collect()
        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE:
        if cfg.TEST.DATASET == 'epickitchens':
            all_preds = (test_meter.verb_video_preds.clone().detach(), test_meter.noun_video_preds.clone().detach())
            all_labels = (test_meter.verb_video_labels.clone().detach(), test_meter.noun_video_labels.clone().detach())
            if cfg.NUM_GPUS:
                all_preds = (all_preds[0].cpu(), all_preds[1].cpu())
                all_labels = (all_labels[0].cpu(), all_labels[1].cpu())
        else:
            all_preds = test_meter.video_preds.clone().detach()
            all_labels = test_meter.video_labels
            if cfg.NUM_GPUS:
                all_preds = all_preds.cpu()
                all_labels = all_labels.cpu()
            
        if writer is not None:
            writer.plot_eval(preds=all_preds, labels=all_labels)

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

            if du.is_root_proc():
                with pathmgr.open(save_path, "wb") as f:
                    pickle.dump([all_preds, all_labels], f)

            logger.info(
                "Successfully saved prediction results to {}".format(save_path)
            )

    test_meter.finalize_metrics()
    return test_meter


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    if len(cfg.TEST.NUM_TEMPORAL_CLIPS) == 0:
        cfg.TEST.NUM_TEMPORAL_CLIPS = [cfg.TEST.NUM_ENSEMBLE_VIEWS]

    test_meters = []
    for num_view in cfg.TEST.NUM_TEMPORAL_CLIPS:

        cfg.TEST.NUM_ENSEMBLE_VIEWS = num_view

        # Build the video model and print model statistics.
        model = build_model(cfg)

        flops, params = 0.0, 0.0
        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            model.eval()
            flops, params = misc.log_model_info(
                model, cfg, use_train_input=False
            )

        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            misc.log_model_info(model, cfg, use_train_input=False)
        if (
            cfg.TASK == "ssl"
            and cfg.MODEL.MODEL_NAME == "ContrastiveModel"
            and cfg.CONTRASTIVE.KNN_ON
        ):
            train_loader = loader.construct_loader(cfg, "train")
            if hasattr(model, "module"):
                model.module.init_knn_labels(train_loader)
            else:
                model.init_knn_labels(train_loader)

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
                patch_func(model.module, prop_attn=cfg.TOME.PROP_ATTN, mode=cfg.TOME.MODE, threshold=cfg.TOME.THRESHOLD, head_aggregation=cfg.TOME.HEAD_AGGREGATION)
                model.module.r = (cfg.TOME.R_VALUE, cfg.TOME.SCHEDULE)
            else:
                if cfg.TOME.LAYER_QUANTITY > 1:
                    cfg.TOME.R_VALUE = [0] * cfg.TOME.LAYER_TO_DUPLICATE + [cfg.TOME.R_VALUE] * cfg.TOME.LAYER_QUANTITY + [0] * (11 - cfg.TOME.LAYER_TO_DUPLICATE)
                    duplicate_func(model, layer_to_duplicate=cfg.TOME.LAYER_TO_DUPLICATE, quantity=cfg.TOME.LAYER_QUANTITY)
                patch_func(model, prop_attn=cfg.TOME.PROP_ATTN, mode=cfg.TOME.MODE, threshold=cfg.TOME.THRESHOLD, head_aggregation=cfg.TOME.HEAD_AGGREGATION)
                model.r = (cfg.TOME.R_VALUE, cfg.TOME.SCHEDULE)

        # Create video testing loaders.
        test_loader = loader.construct_loader(cfg, "test")
        logger.info("Testing model for {} iterations".format(len(test_loader)))

        if cfg.DETECTION.ENABLE:
            assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
            test_meter = AVAMeter(len(test_loader), cfg, mode="test")
        else:
            assert (
                test_loader.dataset.num_videos
                % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
                == 0
            )
            # Create meters for multi-view testing.
            if cfg.TEST.DATASET == 'epickitchens':
                test_meter = EPICTestMeter(
                    len(test_loader.dataset)
                    // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
                    cfg,
                    cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
                    [97, 300],
                    len(test_loader),
                    cfg.WANDB.ENABLE
                )
            else:
                test_meter = TestMeter(
                    test_loader.dataset.num_videos
                    // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
                    cfg,
                    cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
                    cfg.MODEL.NUM_CLASSES
                    if not cfg.TASK == "ssl"
                    else cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM,
                    len(test_loader),
                    cfg.DATA.MULTI_LABEL,
                    cfg.DATA.ENSEMBLE_METHOD,
                    cfg.WANDB.ENABLE
                )

        # Set up writer for logging to Tensorboard format.
        if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
            cfg.NUM_GPUS * cfg.NUM_SHARDS
        ):
            writer = tb.TensorboardWriter(cfg)
        else:
            writer = None

        if cfg.WANDB.ENABLE and du.is_master_proc(
            cfg.NUM_GPUS * cfg.NUM_SHARDS
        ):  
            wandb.init(project=f'{cfg.WANDB.PROJECT}', config=cfg)

        # # Perform multi-view test on the entire dataset.
        test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
        test_meters.append(test_meter)
        if writer is not None:
            writer.close()

    result_string_views = "_p{:.2f}_f{:.2f}".format(params / 1e6, flops)

    for view, test_meter in zip(cfg.TEST.NUM_TEMPORAL_CLIPS, test_meters):
        logger.info(
            "Finalized testing with {} temporal clips and {} spatial crops".format(
                view, cfg.TEST.NUM_SPATIAL_CROPS
            )
        )
        if cfg.TEST.DATASET == 'epickitchens':
            result_string_views += "_{}a_action{}" "".format(
                view, test_meter.stats["action_top1_acc"]
            )
            result_string_views += "_{}a_noun{}" "".format(
                view, test_meter.stats["noun_top1_acc"]
            )
            result_string_views += "_{}a_verb{}" "".format(
                view, test_meter.stats["verb_top1_acc"]
            )

            result_string = (
                "_p{:.2f}_f{:.2f}_{}a_action{}_a_noun{}_ a_verb{} Action Top5 Acc: {}  Noun Top5 Acc {} Verb Top5 Acc {} MEM: {:.2f} f: {:.4f}"
                "".format(
                    params / 1e6,
                    flops,
                    view,
                    test_meter.stats["action_top1_acc"],
                    test_meter.stats["noun_top1_acc"],
                    test_meter.stats["verb_top1_acc"],
                    test_meter.stats["action_top5_acc"],
                    test_meter.stats["noun_top5_acc"],
                    test_meter.stats["verb_top5_acc"],
                    misc.gpu_mem_usage(),
                    flops,
                )
            )
        else:
            result_string_views += "_{}a{}" "".format(
                view, test_meter.stats["top1_acc"]
            )

            result_string = (
                "_p{:.2f}_f{:.2f}_{}a{} Top5 Acc: {} MEM: {:.2f} f: {:.4f}"
                "".format(
                    params / 1e6,
                    flops,
                    view,
                    test_meter.stats["top1_acc"],
                    test_meter.stats["top5_acc"],
                    misc.gpu_mem_usage(),
                    flops,
                )
            )

        logger.info("{}".format(result_string))
    logger.info("{}".format(result_string_views))
    if cfg.WANDB.ENABLE and du.is_master_proc(
            cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        wandb.finish()
    return result_string + " \n " + result_string_views
