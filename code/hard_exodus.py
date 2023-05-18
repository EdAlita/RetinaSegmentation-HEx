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

    result = []
    result2 = []
    
    data_length = len(data)
    
    #log_fuction
    hardexodus_logger = projloggger('hardexodus',timestamp,dataname,loglevel,'tmp25')
    hardexodus_logger.debug("Begin of the hard_exodus.py code")
    
    kernel = np.ones((5,5),np.uint8)
    strutElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    strutElement2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,10))
    
    def hardExodus(img):
        zeros = np.zeros((2848,4288),dtype=np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        histogram=cv2.calcHist(img, [0], None, [256],(0,255), accumulate=False)
        _, tresh = cv2.threshold(img,np.percentile(img,95),255,cv2.THRESH_BINARY)
        Opening = cv2.morphologyEx(tresh,cv2.MORPH_OPEN,kernel,iterations=1)
        Closing = cv2.morphologyEx(Opening,cv2.MORPH_CLOSE,kernel,iterations=1)
        
        contours, hierarchy  = cv2.findContours(Closing,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        largest_countour = sorted(contours,key=cv2.contourArea)
        for cnt in range(0,len(largest_countour)):
            cv2.drawContours(zeros,[largest_countour[cnt]],-1,(255,255,255),-1)
            
        return Closing
    
    
    with tqdm(total=data_length,desc="Hard Exodus Extraction "+dataname) as pbar:
        for i in range(0,data_length):
            result.append(hardExodus(data[i]))
            hardexodus_logger.info("Hard Exodus of "+str(i)+" of"+str(dataname))
            pbar.update(1)
    

        
    hardexodus_logger.debug("The code run was sucessful")
    hardexodus_logger.debug("exit code 0")
    
    masks_data = result
    return result