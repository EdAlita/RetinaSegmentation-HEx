from glob import glob
import cv2
import logging
import time
import sys
import os
from extendable_logger import extendable_logger
from matplotlib import pyplot as plt
from time import sleep
from tqdm import tqdm
from PIL import Image
import numpy as np


def mask(timestr,trash,dname,data,intermedateResult=None):
    img = np.zeros((2848, 4288, 3), dtype = "uint8")
    masks = []
    
    if (trash!=0):
        mask_logger = extendable_logger('mask',"logs/"+timestr+"/"+dname+"mask.log",level=trash)
    else:
        mask_logger = extendable_logger('main',"tmp3",trash)
        mask_logger.disabled = True
        os.remove("tmp3")
    mask_logger.debug("Begin of the mask.py code")   
    
    #Eye fundus Mask
    with tqdm(total=len(data),desc="Mask of "+dname) as pbar:
       for i in range(0,len(data)):
           img = data[i]
           (G, R, B) = cv2.split(img)
           I_q = R/(G+1)
           I_q = cv2.medianBlur(data[i], 5)
           gs = cv2.cvtColor(data[i], cv2.COLOR_BGR2GRAY)
           ret2,th2 = cv2.threshold(gs,0,250,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
           data[i] = (I_q[:,:,0] > th2) * 255
           data[i] = (255-data[i])
           pbar.update(1)

    if(intermedateResult!=None):
        row = intermedateResult%2 # Get subplot row
        column =  intermedateResult//2 # Get subplot column
        for i in range(0,len(data)):
            plt.subplot(3,4,i+1)    # the number of images in the grid is 5*5 (25)
            plt.imshow(data[i],cmap='gray')
        plt.show()
    
    mask_logger.debug("The code run was sucessful")
    mask_logger.debug("exit code 0")
    
    maskpos_data = data
    return maskpos_data    