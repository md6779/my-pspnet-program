from importlib.resources import path

#from (root directory) import (py file)
import argparse
import threading
from threading import Thread
import logging
import time 
import concurrent.futures

from utils import *

def thread_function(name):
    logging.info("Thread %s: starting", name)
    time.sleep(2)
    logging.info("Thread %s: finishing", name)

def main():
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=3
        ) as executor:
        executor.map(thread_function, range(3))

# def many_threads():
#     threads = list()
#     for index in range(3):
#         logging.info("Main    : create and start thread %d.", 
#         index)
#         x = threading.Thread(target=thread_function, 
#         args=(index,))
#         threads.append(x)
#         x.start()

#     for index, thread in enumerate(threads):
#         logging.info("Main    : before joining thread %d.", 
#         index)
#         thread.join()
#         logging.info("Main    : thread %d done", index)

if __name__ == "__main__":
    main()