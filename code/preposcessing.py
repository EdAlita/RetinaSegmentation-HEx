from glob import glob
import cv2
import logging
import time
import sys
import os
from extendable_logger import extendable_logger


def prepos(timestr,trash):
    pre_logger = extendable_logger('preposcessing',"logs/"+timestr+"/prepos.log",level=trash)
    pre_logger.debug("Begin of the prepos.py code")
    
    pre_logger.debug("The code run was sucessful")
    pre_logger.debug("exit code 0")