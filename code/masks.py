from glob import glob
import cv2
import logging
import time
import sys
import os
from extendable_logger import extendable_logger
from extendable_logger import projloggger
from matplotlib import pyplot as plt
from time import sleep
from tqdm import tqdm
import numpy as np

def masks(timestr,trash,dname,data,intermedateResult=0):
    img = np.zeros((2848, 4288, 3), dtype = "uint8")
    tresh = []
    imCopy= []
    
    #log_fuction
    masks_logger = projloggger('masks',timestr,dname,trash,'tmp4')
    masks_logger.debug("Begin of the masks.py code")
    
    def createMask(rows,cols,img):
        out = np.zeros((2848, 4288, 1), dtype = "uint8")
        imgray = np.zeros((2848, 4288, 1), dtype = "uint8")
        # black image
        imCopy = img.copy()
        # blit our contours onto it in white color
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out = cv2.medianBlur(imgray,13)
        ret,thresh = cv2.threshold(imgray,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(imCopy,contours,-1, (0,255,255))
        return thresh,imCopy
    
    with tqdm(total=len(data),desc="Masks Extraction "+dname) as pbar:
        for i in range(0,len(data)):
            bus1,bus2 = createMask(data[i].shape[0],data[i].shape[0], data[i])
            masks_logger.info("Creating Masks of "+str(i)+" of"+str(dname))
            tresh.append(bus1)
            imCopy.append(bus2)
            pbar.update(1)
    

    
    #printing intermedate results
    if(intermedateResult!=0):
        
        plt.subplot(121),plt.imshow(cv2.cvtColor(data[intermedateResult], cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.subplot(122),plt.imshow(cv2.cvtColor(tresh[intermedateResult], cv2.COLOR_BGR2RGB))
        plt.title("Mask Image")
        
        plt.savefig("logs/"+timestr+"/masksResults"+str(intermedateResult)+".pdf")
        
    masks_logger.debug("The code run was sucessful")
    masks_logger.debug("exit code 0")
    
    masks_data = tresh
    return masks_data 