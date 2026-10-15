"""Run the isolated SAFE-SILENCE scientific extension from any working directory."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from safe_silence.evaluation.extension import main

if __name__ == "__main__":
    main([*sys.argv[1:], "--root", str(PROJECT_ROOT)])
