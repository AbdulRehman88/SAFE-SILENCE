#!/usr/bin/env python3
"""Run Task J causal semantic innovations and bounded-validity decisions."""

from safe_silence.cli import main

if __name__ == "__main__":
    main(["build-semantic-evidence", "--config", "configs/pipeline.yaml"])
