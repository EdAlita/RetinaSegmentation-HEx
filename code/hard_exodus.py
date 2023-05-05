import cv2
from extendable_logger import projloggger
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from proj_functions import transformRGB2YIQ,transformYIQ2RGB

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
    
    data_length = len(data)
    
    #log_fuction
    hardexodus_logger = projloggger('hardexodus',timestamp,dataname,loglevel,'tmp25')
    hardexodus_logger.debug("Begin of the hard_exodus.py code")
    
    kernel = np.ones((5,5),np.uint8)
    strutElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
    strutElement2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
    
    def hardExodus(img):
        
        dilateImage = cv2.morphologyEx(img,cv2.MORPH_CLOSE,strutElement)
        result = cv2.morphologyEx(img,cv2.MORPH_CLOSE,strutElement2)
        
        retValue, treshImage = cv2.threshold(img,125,255, cv2.THRESH_BINARY)
        I3 = cv2.bitwise_and(dilateImage,treshImage)
        I4 = cv2.bitwise_and(result,treshImage)
        
        I5 = cv2.bitwise_xor(dilateImage,result)
        I6 = cv2.bitwise_and(dilateImage,result)
        
        
        
        Iresult = cv2.add(I3,I4)

        
        
        
        """
        Z = img.reshape((-1,3))
        
        Z = np.float32(Z)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10, 1.0)
        K = 8
        ret, label, center = cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        """

        
        #retValue, treshImage = cv2.threshold(gray,150,193, cv2.THRESH_BINARY)
        #result = cv2.morphologyEx(treshImage,cv2.MORPH_OPEN,strutElement2)
        
        #result = cv2.bitwise_or(dilateImage,result)
        #result = cv2.dilate(result,kernel)

        #medianblur = cv2.erode(treshImage,kernel)
        return Iresult
    
    
    with tqdm(total=data_length,desc="Hard Exodus Extraction "+dataname) as pbar:
        for i in range(0,data_length):
            result.append(hardExodus(data[i]))
            hardexodus_logger.info("Hard Exodus of "+str(i)+" of"+str(dataname))
            pbar.update(1)
    
        
    hardexodus_logger.debug("The code run was sucessful")
    hardexodus_logger.debug("exit code 0")
    
    masks_data = result
    return masks_data 