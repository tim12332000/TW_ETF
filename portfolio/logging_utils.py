import re
import sys


class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")
        self.ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        self.log_filtered = False

    def write(self, message):
        self.terminal.write(message)

        if "[CACHE]" in message or "[CAL]" in message:
            self.log_filtered = True
            return

        if message == "\n" and self.log_filtered:
            self.log_filtered = False
            return

        self.log_filtered = False
        clean_message = self.ansi_escape.sub("", message)
        self.log.write(clean_message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.flush()
        if self.log and not self.log.closed:
            self.log.close()
