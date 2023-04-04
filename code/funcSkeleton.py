#Imports for the functionality
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

#    - timestr : string of the time and date of running for saving logs in the folder and Intermadate Results
#    - trash   : Value for activating of deactivating the log capabilities.
#    - dname   : Name of the data that we are applying the fuction
#    - data    : The data that the function is going to work with
#    - intermedateResult : Activated the priting of the results. (Only works with activation of logs at the same time)

def function_name(timestr,trash,dname,data,intermedateResult=None):
    
    #log_fuction
    if (trash!=0):
        pre_logger = extendable_logger('function_name',"logs/"+timestr+"/"+dname+"function_name.log",level=trash)
    else:
        #Create a different tmp number just for deleting so number no matters just not repeting
        pre_logger = extendable_logger('main',"tmp3",trash)
        pre_logger.disabled = True
        os.remove("tmp3")
    pre_logger.debug("Begin of the function_name.py code")
    
    with tqdm(total=len(data)-1,desc="Part of the process"+dname) as pbar:
        for i in range(0,len(data)):
            #the code run in this section will appear as a progress bar when run in cmd line
            pbar.update(1) #udating the progress in the bar
    
 
    #Saving Intermedate Results on a PDF for the value save in intermedateResult
    if(intermedateResult!=None):
        
        plt.subplot(131),plt.imshow(cv2.cvtColor(data[intermedateResult], cv2.COLOR_BGR2RGB))
        plt.title("R1")
        plt.subplot(132),plt.imshow(cv2.cvtColor(cli[intermedateResult], cv2.COLOR_BGR2RGB))
        plt.title("R2")
        plt.subplot(133),plt.imshow(cv2.cvtColor(bus[intermedateResult], cv2.COLOR_BGR2RGB))
        plt.title("R3")
        #It save them on the logs folder
        plt.savefig("logs/"+timestr+"/FunctionNameResults"+str(intermedateResult)+".pdf")
    
    #Whit this we know that the code run until the end    
    pre_logger.debug("The code run was sucessful")
    pre_logger.debug("exit code 0")
    
    #return the data list of all images
    prepos_data = bus
    return prepos_data 