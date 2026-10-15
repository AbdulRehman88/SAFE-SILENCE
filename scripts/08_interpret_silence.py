#!/usr/bin/env python3
"""Run Task L receiver-side dual belief and bounded-silence interpretation."""

from safe_silence.cli import main

if __name__ == "__main__":
    main(["interpret-silence", "--config", "configs/pipeline.yaml"])
