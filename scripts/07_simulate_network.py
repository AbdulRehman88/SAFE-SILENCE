#!/usr/bin/env python3
"""Run Task K matched acoustic-network realizations and transport evaluation."""

from safe_silence.cli import main

if __name__ == "__main__":
    main(["simulate-network", "--config", "configs/pipeline.yaml"])
