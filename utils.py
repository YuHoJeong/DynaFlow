from torch.utils.tensorboard import SummaryWriter
import os
import time
from tensorboard import program
import webbrowser
import shutil

class Logger:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def scalar(self, tag: str, value, step: int):
        self.writer.add_scalar(tag, value, step)

    def close(self):
        self.writer.close()


class Timer:
    def __init__(self):
        self._start = time.time()
    def __call__(self, reset=True):
        now = time.time()
        diff = now - self._start
        if reset:
            self._start = now
        return diff



def find_free_port(start_port=6006):
    import socket
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
            port += 1


def save_code_snapshot(source_files, save_dir):
    snapshot_dir = os.path.join(save_dir, 'snapshot')
    os.makedirs(snapshot_dir, exist_ok=True)

    print("\n" + "-" * 20 + " Saving Code Snapshot " + "-" * 20)
    for file_path in source_files:
        shutil.copy(file_path, snapshot_dir)
    print("-" * 64 + "\n")



if "__main__" == __name__:
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', data_path, '--port', str(find_free_port(6006))])
    url = tb.launch()
    print(f"[DynaFlow] Tensorboard: {url}")
    time.sleep(100000)