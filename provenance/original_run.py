from pathlib import Path
import sys
import warnings

sys.path.insert(0, '/Users/6t4/Desktop/d2nn/src')
from multiwavelenth_donn.training_pipeline import run_local_training_job


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.stream.write(value)
        self.log.write(value)
        self.log.flush()

    def flush(self):
        self.stream.flush()
        self.log.flush()


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=FutureWarning, message='.*torch.cuda.amp.autocast.*')
    folder = Path(__file__).resolve().parent
    with (folder / 'training.log').open('w') as log:
        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, log)
        try:
            result = run_local_training_job(config_path=folder / 'config.yaml', project_root=Path('/Users/6t4/Desktop/d2nn'))
            print('FINISHED', result['job_dir'])
        finally:
            sys.stdout.flush()
            sys.stdout = original_stdout
