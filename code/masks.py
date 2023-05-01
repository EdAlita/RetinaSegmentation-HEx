import cv2
from extendable_logger import projloggger
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np

def masks(timestamp,loglevel,dataname,data,intermedateResult=0):
    """Function to get the mask of the images given

    Args:
        timestr (str): Day and time to create the logs of the project
        loglevel (int): Level of debuging for the logs
        dataname (str): Name of the data for the logs
        data (List): List of data to apply the preprocessing 
        intermedateResult (int, optional): The number of intermedate result to save in pdf format of the results. Defaults to 0.

    Returns:
        img_array: Result of the mask image function
    """
    img = np.zeros((2848, 4288, 3), dtype = "uint8")
    tresh = []
    imageCopy= []
    
    data_length = len(data)
    
    #log_fuction
    masks_logger = projloggger('masks',timestamp,dataname,loglevel,'tmp4')
    masks_logger.debug("Begin of the masks.py code")
    
    def createMask(img):
        """create a mask of rows by cols of image

        Args:
            img (img): Image of creating the mask

        Returns:
            img: mask of the image
        """
        imgray = np.zeros((2848, 4288, 1), dtype = "uint8")
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret,tresh = cv2.threshold(imgray,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return tresh
    
    with tqdm(total=data_length,desc="Masks Extraction "+dataname) as pbar:
        for i in range(0,data_length):
            tresh.append(createMask(data[i]))
            masks_logger.info("Creating Masks of "+str(i)+" of"+str(dataname))
            pbar.update(1)
    
    #printing intermedate results
    if(intermedateResult!=0):
        
        plt.subplot(121),plt.imshow(cv2.cvtColor(data[intermedateResult], cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.subplot(122),plt.imshow(cv2.cvtColor(tresh[intermedateResult], cv2.COLOR_BGR2RGB))
        plt.title("Mask Image")
        
        plt.savefig("logs/"+timestamp+"/masksResults"+str(intermedateResult)+".pdf")
        
    masks_logger.debug("The code run was sucessful")
    masks_logger.debug("exit code 0")
    
    masks_data = tresh
    return masks_data 