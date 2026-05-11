import sys
import logging

DEFAULT_LOG_FILE = "stt-error.log"
DEFAULT_NAME = "stt"
LOG_FMT = "%(asctime)s %(levelname)s %(message)s"
TIME_FMT = "%H:%M:%S"


class STTLogger:
    """STT service logger."""

    def __init__(self, name=DEFAULT_NAME, log_file=DEFAULT_LOG_FILE, debug=False):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

        level = logging.DEBUG if debug else logging.INFO

        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)

        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.ERROR)

        fmt = logging.Formatter(LOG_FMT, TIME_FMT)
        ch.setFormatter(fmt)
        fh.setFormatter(fmt)

        if not self.logger.hasHandlers():
            self.logger.addHandler(ch)
            self.logger.addHandler(fh)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        self.logger.exception(msg, *args, **kwargs)
