import cv2
from extendable_logger import projloggger
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np

def opticaldisk(timestamp,loglevel,dataname,data):
    """Function to extract the optical disk

    Args:
        timestr (str): Day and time to create the logs of the project
        loglevel (int): Level of debuging for the logs
        dataname (str): Name of the data for the logs
        data (List): List of data to apply the preprocessing 

    Returns:
        img_array: Result of the optical extraction image function
    """
    img = np.zeros((2848, 4288, 3), dtype = "uint8")
    result = []
    
    data_length = len(data)
    
    #log_fuction
    masks_logger = projloggger('masks',timestamp,dataname,loglevel,'tmp4')
    masks_logger.debug("Begin of the masks.py code")
    
    def extractDisk(img):
        """create a dark disk on the optical disk of the image

        Args:
            img (img): Image of creating the mask

        Returns:
            img: mask of the image
        """
        (B, G, R) = cv2.split(img)
        imageBlur = cv2.GaussianBlur(R,(25,25),0)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(imageBlur)
        img = cv2.merge([B, G, R])
        cv2.circle(img,maxLoc, 300, (0,0,0), -1)        
        return img
    
    with tqdm(total=data_length,desc="Optical Disk Extraction "+dataname) as pbar:
        for i in range(0,data_length):
            result.append(extractDisk(data[i]))
            masks_logger.info("optical Disk of "+str(i)+" of"+str(dataname))
            pbar.update(1)
    
        
    masks_logger.debug("The code run was sucessful")
    masks_logger.debug("exit code 0")
    
    masks_data = result
    return masks_data 