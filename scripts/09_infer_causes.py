#!/usr/bin/env python3
"""Run Task M receiver-observable coarse ambiguity-cause inference."""

from safe_silence.cli import main

if __name__ == "__main__":
    main(["infer-causes", "--config", "configs/pipeline.yaml"])
