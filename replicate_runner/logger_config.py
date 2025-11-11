import logging
import os
import gzip
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from queue import Queue

LOG_FILENAME = "replicate_runner.log"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5

def compress_log(src_path):
    """Compress a single log file to .gz and remove the original."""
    if not os.path.exists(src_path):
        return
    gz_path = src_path + ".gz"
    with open(src_path, 'rb') as f_in, gzip.open(gz_path, 'wb') as f_out:
        f_out.writelines(f_in)
    os.remove(src_path)

def create_logger():
    """
    Create a logger that:
      - Handles logs asynchronously via Queue
      - Uses rotating file handler
      - Compresses older backups
    """
    logger = logging.getLogger("replicate_runner")
    logger.setLevel(logging.INFO)

    # Queue for asynchronous logging
    log_queue = Queue(-1)
    queue_handler = QueueHandler(log_queue)
    logger.addHandler(queue_handler)

    # RotatingFileHandler for actual writing
    rotating_handler = RotatingFileHandler(
        LOG_FILENAME,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT
    )
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] - %(message)s"
    )
    rotating_handler.setFormatter(formatter)

    # The "listener" thread that reads from the queue and writes to file
    listener = QueueListener(log_queue, rotating_handler, respect_handler_level=True)
    listener.start()

    # Optional: compress old logs after rotation
    # RotatingFileHandler won't do compression automatically, so you can:
    #   1. Patch or override the doRollover() method,
    #   2. Or run a small routine periodically that compresses older logs,
    #   3. Or handle the compression in a custom rotation logic.
    # For a simpler approach, you can do something like:
    original_doRollover = rotating_handler.doRollover

    def doRolloverWithCompress():
        # rotate
        original_doRollover()
        # after rotate, compress any .1 .2 .3 log if you'd like
        for i in range(1, BACKUP_COUNT + 1):
            old_log = f"{LOG_FILENAME}.{i}"
            if os.path.exists(old_log):
                compress_log(old_log)

    rotating_handler.doRollover = doRolloverWithCompress

    return logger
