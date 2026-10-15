#!/usr/bin/env python3
"""Run Task I causal environmental prediction and uncertainty calibration."""

from safe_silence.cli import main

if __name__ == "__main__":
    main(["train-predictor", "--config", "configs/pipeline.yaml"])
