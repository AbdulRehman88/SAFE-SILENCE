#!/usr/bin/env python3
"""Run Task G leakage-safe OOI harmonization and split construction."""

from safe_silence.cli import main

if __name__ == "__main__":
    main(["preprocess-ooi", "--config", "configs/pipeline.yaml"])
