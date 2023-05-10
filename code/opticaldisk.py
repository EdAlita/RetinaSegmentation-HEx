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
    ye = np.zeros((2848,4288,1), dtype="uint8")
    result = []
    test_correction_number = [11,24]
    test_correct_centers = [(2022,945),(489,942)]
    train_correction_number = [6,9,12,20,24,48]
    train_correct_centers = [(1905,999),(258,501),(1707,807),(513,831),(648,792),(1779,993)]

    data_length = len(data)
    
    #log_fuction
    masks_logger = projloggger('opticaldisk',timestamp,dataname,loglevel,'tmp4')
    masks_logger.debug("Begin of the masks.py code")
    
    def correctionDisk(img,correction_number,correct_centers,number):
        for i in range(0,len(correction_number)):
            if number==correction_number[i]:
                img_circle = cv2.circle(img,correct_centers[i],200,(0,0,0), -1)          
        return img_circle
    
    def extractDisk(img,image_number):

        """Create a dark disk on the optical disk of the image
        
        Args:
            img (img): Image of creating the mask
            
        Returns:
            img: mask of the image
        """
        img_resized = cv2.resize(img,None,fx=0.60,fy=0.60)
        
        (B, G, R) = cv2.split(img_resized)
        
        ye = cv2.addWeighted(R,0.73925,G,0.14675,0)
        ye = cv2.addWeighted(ye,1,B,0.114,0)
        
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(ye)
        img_resized = cv2.merge([B,G,R])
        
        circle = True
        
        if dataname == "Testing":
            if image_number in test_correction_number:
                img_resized = correctionDisk(img_resized,test_correction_number,test_correct_centers,image_number)
                circle = False
                
        if dataname == "Training":    
            if image_number in train_correction_number:
                img_resized = correctionDisk(img_resized,train_correction_number,train_correct_centers,image_number)
                circle = False       
        if (circle):
            cv2.circle(img_resized,maxLoc,200,(0,0,0), -1)
        
        #cv2.selectROI()        
        return img_resized
    
    with tqdm(total=data_length,desc="Optical Disk Extraction "+dataname) as pbar:
        for i in range(0,data_length):
            result.append(extractDisk(data[i],i))
            masks_logger.info("optical Disk of "+str(i)+" of"+str(dataname))
            pbar.update(1)
    
        
    masks_logger.debug("The code run was sucessful")
    masks_logger.debug("exit code 0")
    
    masks_data = result
    return masks_data 