#!/usr/bin/env python3
"""Run the YAML-controlled OOI integrity and coverage audit."""

from safe_silence.cli import main

if __name__ == "__main__":
    main(["audit-ooi", "--config", "configs/pipeline.yaml"])
