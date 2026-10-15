#!/usr/bin/env python3
"""Run Task H leakage-safe, observation-grounded event labeling."""

from safe_silence.cli import main

if __name__ == "__main__":
    main(["label-events", "--config", "configs/pipeline.yaml"])
