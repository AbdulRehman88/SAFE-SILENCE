#!/usr/bin/env python3
"""Validate the canonical SAFE-SILENCE configuration."""

from safe_silence.cli import main

if __name__ == "__main__":
    main(["validate-config", "--config", "configs/pipeline.yaml"])
