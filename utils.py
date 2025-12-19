import warnings
import os
import threading
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Logger:
    def __init__(self, file_path, file_mode="a"):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.file = open(file_path, file_mode, encoding="utf-8")
        self.lock = threading.Lock()  # prevents race conditions

    def __call__(self, message):
        with self.lock:
            self.file.write(f"{message}\n")
            self.file.flush()     # IMPORTANT: flush immediately
            print(message)

    def close(self):
        self.file.close()