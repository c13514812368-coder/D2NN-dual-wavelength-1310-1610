from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / 'run_experiment.py'), 'train'], check=True)
