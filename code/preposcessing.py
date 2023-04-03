from glob import glob
import cv2
import logging
import time
import sys
import os
from extendable_logger import extendable_logger


def prepos(timestr,trash,dname,data,intermedateResult=None):
    bus = data
    if (trash!=0):
        pre_logger = extendable_logger('preposcessing',"logs/"+timestr+"/"+dname+"prepos.log",level=trash)
    else:
        pre_logger = extendable_logger('main',"tmp2",trash)
        pre_logger.disabled = True
        os.remove("tmp2")
    pre_logger.debug("Begin of the prepos.py code")
    
    #Denoising the Images
    for i in range(0,len(data)-1):
        bus[i] = cv2.fastNlMeansDenoisingColored(data[i],None,10,10,7,21)
        pre_logger.info(dname+" image "+str(i)+" Denoising")

    if(intermedateResult!=None):
        plt.subplot(121),plt.imshow(data[intermedateResult])
        plt.subplot(122),plt.imshow(dst[intermedateResult])
        plt.show()
    
    pre_logger.debug("The code run was sucessful")
    pre_logger.debug("exit code 0")
    
    prepos_data = bus
    return prepos_data 