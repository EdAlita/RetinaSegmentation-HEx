import cv2
from tqdm import tqdm
import numpy as np
from loguru import logger

class HardExodus():
    """Class to create the Hard Exodus Segmentation
    """
    
    def __init__(self,
                 dataname,
                 data
                 ):
        """Default initialization
        Args:
            dataname (string): Name of data to analyze
            data (List): List of images
        """
        self.dataname = dataname
        self.hardexodus = data
        self.result = []
        self.result2 = []
        self.kernel = np.ones((2,2),np.uint8)
        self.data_length = len(data)
        logger.info(f"Class Initialized: {self.__class__}")
 
        
    def hardExodus(self,img, threshold):
        """First flow for obtaning the exodus
        Args:
            img (image): image with preprosesing
            threshold (int): integer for the threshold to binarization
        Returns:
            binary_image: binarization of the exodus
        """
        _, tresh = cv2.threshold(img,np.percentile(img,threshold),255,cv2.THRESH_BINARY)
        Opening = cv2.morphologyEx(tresh,cv2.MORPH_OPEN,self.kernel,iterations=1)
        Closing = cv2.morphologyEx(Opening,cv2.MORPH_CLOSE,self.kernel,iterations=1)   
        return Closing
    
    def getHardExodus(self,thresholdList):
        """Get all the exodus of the multy flow
        Args:
            thresholdList (list): two element list to get the exodus
        Returns:
            lists: lists with the multiple results
        """
        with tqdm(total=self.data_length,desc="Hard Exodus Extraction "+self.dataname) as pbar:
            for i in range(0,self.data_length):
                self.result.append(self.hardExodus(self.hardexodus[i],thresholdList[0]))
                self.result2.append(self.hardExodus(self.hardexodus[i],thresholdList[1]))
                pbar.update(1)
                
        return self.result, self.result2
    