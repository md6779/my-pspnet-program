from importlib.resources import path

#from (root directory) import (py file)
import argparse
import threading
from threading import Thread
import logging
import time 
import concurrent.futures

from utils import *

class FakeDatabase:
    def __init__(self):
        self.value = 0
        self._lock = threading.Lock()

    def locked_update(self, name):
        logging.info("Thread %s: starting update", name)
        logging.debug("Thread %s about to lock", name)
        with self._lock:
            logging.debug("Thread %s has lock", name)
            local_copy = self.value
            local_copy += 1
            time.sleep(0.1)
            self.value = local_copy
            logging.debug("Thread %s about to release lock", 
            name)
        logging.debug("Thread %s after release", name)
        logging.info("Thread %s: finishing update", name)

def main():
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    database = FakeDatabase()
    logging.info("Testing update. Starting value is %d.", 
    database.value)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=2
        ) as executor:
        for index in range(2):
            executor.submit(database.locked_update, 
            index)
    logging.info("Testing update. Ending value is %d.", 
    database.value)

def deadlock_ex():
    l = threading.Lock()
    print("before first acquire")
    l.acquire()
    print("before second acquire")
    l.acquire()
    print("acquired lock twice")

if __name__ == "__main__":
#    main()
    deadlock_ex()