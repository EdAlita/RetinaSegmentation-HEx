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

def prepos(timestr,trash,dname,data,intermedateResult=0):
    img = np.zeros((2848, 4288, 3), dtype = "uint8")
    bus = []
    cli= []
    green_ch = []
    
    pre_logger = projloggger('preposcessing',timestr,dname,trash,'tmp2')
    pre_logger.debug("Begin of the prepos.py code")
    
    ### Steps
    ### Gren channel splitting
    ### CLAHE
    ### Denoissing with non Locals
    ### Inverting Image
    
    ### green channel splitting and CLAHE
    pre_logger.debug("Begin of the Preproscessing of "+dname)
    with tqdm(total=len(data),desc="Preposcessing "+dname) as pbar:
        for i in range(0,len(data)):
            img = data[i]
            (B, G, R) = cv2.split(img)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            B = clahe.apply(B)
            G = clahe.apply(G)
            R = clahe.apply(R)
            image_merge = cv2.merge([B, G, R ])
            cli.append(image_merge)
            pre_logger.info(dname+" image "+str(i)+" Preposcessing")
            pbar.update(1)
    pre_logger.debug("End of the Preproscessing of "+dname)
    
    #Denoising the Images
    pre_logger.debug("Begin of the Denoising of "+dname)
    with tqdm(total=len(cli),desc="Denoising of "+dname) as pbar:
        for i in range(0,len(cli)):
            img = cv2.fastNlMeansDenoisingColored(cli[i],None,10,10,7,21)
            img = abs(img - 255)
            bus.append(img)
            pre_logger.info(dname+" image "+str(i)+" Denoising")
            pbar.update(1)
    pre_logger.debug("End of the Denoising of "+dname)
    
    #Printing intermedated Results
    if(intermedateResult!=0):
        pre_logger.debug("Creating intermedated results")
        plt.subplot(131),plt.imshow(cv2.cvtColor(data[intermedateResult], cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.subplot(132),plt.imshow(cv2.cvtColor(cli[intermedateResult], cv2.COLOR_BGR2RGB))
        plt.title("CLAHE")
        plt.subplot(133),plt.imshow(cv2.cvtColor(bus[intermedateResult], cv2.COLOR_BGR2RGB))
        plt.title("Denoising")
        
        plt.savefig("logs/"+timestr+"/PreposcessingResults"+str(intermedateResult)+".pdf")
    
    pre_logger.debug("The code run was sucessful")
    pre_logger.debug("exit code 0")
    
    prepos_data = bus
    return prepos_data 