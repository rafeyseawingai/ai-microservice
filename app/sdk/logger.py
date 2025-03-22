import time
import logging

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Logger:
    @staticmethod
    def log(message):
        logging.info(message)
