"""Generate manuscript figures without rerunning experiments or altering their inputs."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from safe_silence.reporting.confirmation_figures import main

if __name__ == "__main__":
    main(PROJECT_ROOT)
