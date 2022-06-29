from importlib.resources import path

#from (root directory) import (py file)
import argparse
import threading
from threading import Thread
import logging
import time 
import concurrent.futures

from yaml import load
import jpg_to_mp4
import mp4_to_jpg
import load_pth_file

from utils import *

# def thread_function(name):
#     logging.info("Thread %s: starting", name)
#     time.sleep(2)
#     logging.info("Thread %s: finishing", name)

# def main():
#     format = "%(asctime)s: %(message)s"
#     logging.basicConfig(format=format, level=logging.INFO,
#                         datefmt="%H:%M:%S")

#     with concurrent.futures.ThreadPoolExecutor(
#         max_workers=3
#         ) as executor:
#         executor.map(thread_function, range(3))

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

def main():
    
    a = threading.Thread(target = mp4_to_jpg, 
        args=())
    b = threading.Thread(
        target = jpg_to_mp4.png_to_mp4, 
        args=(jpg_to_mp4.png_to_mp4.dir_path_2, 
        jpg_to_mp4.png_to_mp4.name2)
        )
    c = threading.Thread(target=load_pth_file, 
        args=())

    daemon = True
    a.setDaemon(daemon)
    b.setDaemon(daemon)
    c.setDaemon(daemon)

    a.start()
    b.start()
    c.start()

if __name__ == "__main__":
    main()