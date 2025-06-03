#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to benchmark a video classification model."""
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.model_benchmark import model_benchmark
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args


def main():
    """
    Main function to spawn the benchmark process.
    """
    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
        cfg = assert_and_infer_cfg(cfg)

    launch_job(cfg=cfg, init_method=args.init_method, func=model_benchmark)


if __name__ == "__main__":
    main()
