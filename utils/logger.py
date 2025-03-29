#日志打印与保存
import os

class Logger:
    def __init__(self, out_dir):
        self.log_file = os.path.join(out_dir, "log.txt")
        with open(self.log_file, "w") as f:
            f.write("Training Log\n")

    def log(self, message):
        print(message)
        with open(self.log_file, "a") as f:
            f.write(message + "\n")

