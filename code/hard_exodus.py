import cv2
from extendable_logger import projloggger
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np

def hardExodusSegmentation(timestamp,loglevel,dataname,data):
    """Segmentatio of the Hard Exodus in Fundus eye images

    Args:
        timestr (str): Day and time to create the logs of the project
        loglevel (int): Level of debuging for the logs
        dataname (str): Name of the data for the logs
        data (List): List of data to apply the preprocessing 

    Returns:
        img_array: Result of the segementation process
    """
    img = np.zeros((2848, 4288, 3), dtype = "uint8")
    result = []
    
    data_length = len(data)
    
    #log_fuction
    hardexodus_logger = projloggger('hardexodus',timestamp,dataname,loglevel,'tmp25')
    hardexodus_logger.debug("Begin of the hard_exodus.py code")
    
    kernel = np.ones((8,8),np.uint8)
    strutElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    def hardExodus(img):
        dilateImage = cv2.dilate(img,strutElement)
        retValue, treshImage = cv2.threshold(dilateImage,200,255, cv2.THRESH_BINARY)
        medianblur = cv2.erode(treshImage,kernel)
        return medianblur
    
    
    with tqdm(total=data_length,desc="Hard Exodus Extraction "+dataname) as pbar:
        for i in range(0,data_length):
            result.append(hardExodus(data[i]))
            hardexodus_logger.info("Hard Exodus of "+str(i)+" of"+str(dataname))
            pbar.update(1)
    
        
    hardexodus_logger.debug("The code run was sucessful")
    hardexodus_logger.debug("exit code 0")
    
    masks_data = result
    return masks_data 