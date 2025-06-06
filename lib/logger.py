import os
import logging
import time
from datetime import datetime

def get_logger(root, name=None, debug=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if not debug:
        logfile = os.path.join(root, 'run.log')
        file_handler = logging.FileHandler(logfile, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def train_model(epochs, logger):
    for epoch in range(1, epochs + 1):
        start_time = time.time()  # Start timing the epoch

        # Simulate training process
        logger.info(f"Starting Epoch {epoch}")
        time.sleep(1)  # Simulating time taken per epoch (replace with actual training code)

        # End timing
        end_time = time.time()
        duration = end_time - start_time

        # Log the duration
        logger.info(f"Epoch {epoch} completed in {duration:.2f} seconds")

if __name__ == '__main__':
    time_str = datetime.now().strftime('%Y%m%d%H%M%S')
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)

    logger = get_logger(log_dir, debug=True)
    logger.info("Logger initialized.")

    # Simulate training
    train_model(epochs=5, logger=logger)
