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
import numpy as np

def veins(timestr,trash,dname,data,intermedateResult=None):
    
    #log_fuction
    if (trash!=0):
        pre_logger = extendable_logger('veins',"logs/"+timestr+"/"+dname+"veins.log",level=trash)
    else:
        pre_logger = extendable_logger('main',"tmp3",trash)
        pre_logger.disabled = True
        os.remove("tmp3")
    pre_logger.debug("Begin of the veins.py code")
    
    with tqdm(total=len(data)-1,desc="Veins Extraction "+dname) as pbar:
        for i in range(0,len(data)):

            pbar.update(1)
    

    
    #printing intermedate results
    if(intermedateResult!=None):
        
        plt.subplot(131),plt.imshow(cv2.cvtColor(data[intermedateResult], cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.subplot(132),plt.imshow(cv2.cvtColor(cli[intermedateResult], cv2.COLOR_BGR2RGB))
        plt.title("CLAHE")
        plt.subplot(133),plt.imshow(cv2.cvtColor(bus[intermedateResult], cv2.COLOR_BGR2RGB))
        plt.title("Denoising")
        
        plt.savefig("logs/"+timestr+"/VeinsResults"+str(intermedateResult)+".pdf")
        
    pre_logger.debug("The code run was sucessful")
    pre_logger.debug("exit code 0")
    
    prepos_data = bus
    return prepos_data 