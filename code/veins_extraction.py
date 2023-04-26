from glob import glob
import cv2
from extendable_logger import projloggger
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np

def veins(timestr,trash,dname,data,intermedateResult=0):
    
    #log_fuction
    veins_logger = projloggger('veins',timestr,dname,trash,'tmp3')
    veins_logger.debug("Begin of the veins.py code")
    
    with tqdm(total=len(data)-1,desc="Veins Extraction "+dname) as pbar:
        for i in range(0,len(data)):
            
            pbar.update(1)
    

    
    #printing intermedate results
    if(intermedateResult!=0):
        
        plt.subplot(131),plt.imshow(cv2.cvtColor(data[intermedateResult], cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.subplot(132),plt.imshow(cv2.cvtColor(cli[intermedateResult], cv2.COLOR_BGR2RGB))
        plt.title("CLAHE")
        plt.subplot(133),plt.imshow(cv2.cvtColor(bus[intermedateResult], cv2.COLOR_BGR2RGB))
        plt.title("Denoising")
        
        plt.savefig("logs/"+timestr+"/VeinsResults"+str(intermedateResult)+".pdf")
        
    veins_logger.debug("The code run was sucessful")
    veins_logger.debug("exit code 0")
    
    veins_data = bus
    return veins_data 