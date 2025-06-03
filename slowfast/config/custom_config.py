#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""
from fvcore.common.config import CfgNode

def add_custom_config(_C):
    # -----------------------------------------------------------------------------
    # TimeSformer options
    # -----------------------------------------------------------------------------
    _C.TIMESFORMER = CfgNode()

    _C.TIMESFORMER.ATTENTION_TYPE = 'divided_space_time'

    _C.TIMESFORMER.PRETRAINED_MODEL = ''

    # ---------------------------------------------------------------------------- 
    # Motionformer options
    # ---------------------------------------------------------------------------- 
    _C.MOTIONFORMER = CfgNode()

    # Patch-size spatial to tokenize input
    _C.MOTIONFORMER.PATCH_SIZE = 16

    # Patch-size temporal to tokenize input
    _C.MOTIONFORMER.PATCH_SIZE_TEMP = 2

    # Number of input channels
    _C.MOTIONFORMER.CHANNELS = 3

    # Embedding dimension
    _C.MOTIONFORMER.EMBED_DIM = 768

    # Depth of transformer: number of layers
    _C.MOTIONFORMER.DEPTH = 12

    # number of attention heads
    _C.MOTIONFORMER.NUM_HEADS = 12

    # expansion ratio for MLP
    _C.MOTIONFORMER.MLP_RATIO = 4

    # add bias to QKV projection layer
    _C.MOTIONFORMER.QKV_BIAS = True

    # video input
    _C.MOTIONFORMER.VIDEO_INPUT = True

    # temporal resolution i.e. number of frames
    _C.MOTIONFORMER.TEMPORAL_RESOLUTION = 8

    # use MLP classification head
    _C.MOTIONFORMER.USE_MLP = False

    # Dropout rate for
    _C.MOTIONFORMER.DROP = 0.0

    # Stochastic drop rate
    _C.MOTIONFORMER.DROP_PATH = 0.0

    # Dropout rate for MLP head
    _C.MOTIONFORMER.HEAD_DROPOUT = 0.0

    # Dropout rate for positional embeddings
    _C.MOTIONFORMER.POS_DROPOUT = 0.0

    # Dropout rate 
    _C.MOTIONFORMER.ATTN_DROPOUT = 0.0

    # Activation for head
    _C.MOTIONFORMER.HEAD_ACT = "tanh"

    # Use IM pretrained weights
    _C.MOTIONFORMER.IM_PRETRAINED = True

    # Pretrained weights type
    _C.MOTIONFORMER.PRETRAINED_WEIGHTS = "MOTIONFORMER_1k"

    # Type of position embedding
    _C.MOTIONFORMER.POS_EMBED = "separate"

    # Self-Attention layer
    _C.MOTIONFORMER.ATTN_LAYER = "trajectory"

    # Flag to use original trajectory attn code
    _C.MOTIONFORMER.USE_ORIGINAL_TRAJ_ATTN_CODE = True

    # Approximation type
    _C.MOTIONFORMER.APPROX_ATTN_TYPE = "none"

    # Approximation Dimension
    _C.MOTIONFORMER.APPROX_ATTN_DIM = 128

    # ---------------------------------------------------------------------------- 
    # ViViT options
    # ---------------------------------------------------------------------------- 
    _C.VIVIT = CfgNode()

    # Path to ViViT config
    _C.VIVIT.CONFIG_PATH = ''

    # ---------------------------------------------------------------------------- 
    # VideoMAE options
    # ---------------------------------------------------------------------------- 
    _C.VIDEOMAE = CfgNode()

    # VideoMAE model to build
    _C.VIDEOMAE.MODEL = 'vit_small_patch16_224'

    # Depth of tubelets
    _C.VIDEOMAE.TUBELET_SIZE = 2

    # Fully connected dropout
    _C.VIDEOMAE.FC_DROP_RATE = 0.0

    # Intermediate dropout
    _C.VIDEOMAE.DROP_RATE = 0.0

    # Drop path rate
    _C.VIDEOMAE.DROP_PATH_RATE = 0.1

    # Attention dropout
    _C.VIDEOMAE.ATTN_DROP_RATE = 0.0

    # Mean pooling instead of cls_token
    _C.VIDEOMAE.USE_MEAN_POOLING = True

    # Unsure
    _C.VIDEOMAE.INIT_SCALE = 0.001

    # ----------------------------------------------------------------------------
    # wandb options
    # ----------------------------------------------------------------------------
    _C.WANDB = CfgNode()
    
    # Enable logging to wandb
    _C.WANDB.ENABLE = False

    # Project to log to
    _C.WANDB.PROJECT = ''

    # ----------------------------------------------------------------------------
    # ToMe options
    # ----------------------------------------------------------------------------
    _C.TOME = CfgNode()

    # Patch model
    _C.TOME.ENABLE = False

    # Number of tokens to merge per layer
    _C.TOME.R_VALUE = 0

    # Schedule for r, includes 0 for constant, -1 for decreasing, 1 for increasing
    _C.TOME.SCHEDULE = 0

    # Whether proportional attention enabled
    _C.TOME.PROP_ATTN = True

    # How the similarity metric is aggregated across heads
    _C.TOME.HEAD_AGGREGATION = 'mean'

    # The way that redundant tokens are handled, includes 'merge', 'random_merge', 'drop', 'random_drop', 'hybrid'
    _C.TOME.MODE = 'merge'

    # Threshold that destination tokens are dropped under
    _C.TOME.THRESHOLD = -1.0

    # Index of transformer layer to duplicate
    _C.TOME.LAYER_TO_DUPLICATE = 0

    # Total number of instances of this layer wanted in model (including original instance)
    _C.TOME.LAYER_QUANTITY = 1

    # -----------------------------------------------------------------------------
    # EPIC-KITCHENS options
    # -----------------------------------------------------------------------------
    _C.EPICKITCHENS = CfgNode()

    # Path to Epic-Kitchens RGB data directory
    _C.EPICKITCHENS.VISUAL_DATA_DIR = ''

    # Path to Epic-Kitchens Annotation directory
    _C.EPICKITCHENS.ANNOTATIONS_DIR = ''

    # List of EPIC-100 TRAIN files
    _C.EPICKITCHENS.TRAIN_LIST = 'EPIC_100_train.pkl'

    # List of EPIC-100 VAL files
    _C.EPICKITCHENS.VAL_LIST = 'EPIC_100_validation.pkl'

    # List of EPIC-100 TEST files
    _C.EPICKITCHENS.TEST_LIST = 'EPIC_100_validation.pkl'

    # Testing split
    _C.EPICKITCHENS.TEST_SPLIT = 'validation'

    # Use Train + Val
    _C.EPICKITCHENS.TRAIN_PLUS_VAL = False

    # Number of classes
    _C.EPICKITCHENS.NUM_CLASSES = None

    # Directory structure of data
    _C.EPICKITCHENS.PARTICIPANT_FIRST = True

    # -----------------------------------------------------------------------------
    # Optimizer options
    # -----------------------------------------------------------------------------

    # Number of batches to accumulate loss for
    _C.SOLVER.ACCUMULATE_STEPS = 1

    # -----------------------------------------------------------------------------
    # Dataset utils options
    # -----------------------------------------------------------------------------
    _C.DATASET_UTILS = CfgNode()

    # Create useable versions of the datasets with one entry per class for testing
    _C.DATASET_UTILS.CREATE_MINI_DATASETS = False

    # Save output model probabilities for each dataset element
    _C.DATASET_UTILS.CACHE_MODEL_PROBS = False

    # Save KL divergences between dataset elements
    _C.DATASET_UTILS.CACHE_KL_DIVERGENCES = False

    # Save lengths of video files (K400 or SSv2)
    _C.DATASET_UTILS.SAVE_LENGTHS = False

    # Path to location on disk to store probs and divergences
    _C.DATASET_UTILS.DISTRIBUTION_PATH = ''

    # -----------------------------------------------------------------------------
    # Model benchmark options
    # -----------------------------------------------------------------------------
    _C.MODEL_BENCHMARK = CfgNode()

    # Number of batches to warmup the GPU with
    _C.MODEL_BENCHMARK.WARMUP_ITERATIONS = 0 

    # Number of batches to pass through the benchmark and average over
    _C.MODEL_BENCHMARK.ITERATIONS = 0

    # -----------------------------------------------------------------------------
    # Testing options
    # -----------------------------------------------------------------------------

    _C.TEST.CLIP_LENGTH_HISTOGRAM = False