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
        self.result3 = []
        self.kernel = np.ones((2,2),np.uint8)
        self.data_length = len(data)
        self.kernel_size = 30
        logger.info(f"Class Initialized: {self.__class__}")
        
        
    def hardExodus2(self, img, th):
        """Creates the Segmentation for the small exodus

        Args:
            img (image): Image to analyze from prepocessing
            th (int): treshold for binarizing

        Returns:
            binary_image: binarization of the exodus
        """

        # Create kernels for different directions
        kernel_horizontal = np.zeros((self.kernel_size, self.kernel_size), dtype=np.uint8)
        kernel_horizontal[int(self.kernel_size/2), :] = 1

        kernel_vertical = np.zeros((self.kernel_size, self.kernel_size), dtype=np.uint8)
        kernel_vertical[:, int(self.kernel_size/2)] = 1

        kernel_diagonal_ne_sw = np.eye(self.kernel_size, dtype=np.uint8)
        kernel_diagonal_nw_se = np.fliplr(kernel_diagonal_ne_sw)

        # Perform top-hat operation in different directions
        tophat_horizontal = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel_horizontal)
        tophat_vertical = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel_vertical)
        tophat_diagonal_ne_sw = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel_diagonal_ne_sw)
        tophat_diagonal_nw_se = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel_diagonal_nw_se)

        # Combine the top-hat results using bitwise OR
        final_result = cv2.bitwise_or(cv2.bitwise_or(tophat_horizontal, tophat_vertical),cv2.bitwise_or(tophat_diagonal_ne_sw, tophat_diagonal_nw_se))
        opening = cv2.morphologyEx(final_result, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
        _, tresh = cv2.threshold(opening,np.percentile(opening,th),255,cv2.THRESH_BINARY)
        
        return tresh

        
    def hardExodus(self,img, threshold, kernel):
        """Creates the Segmentation for the big exodus
        Args:
            img (image): image with preprosesing
            threshold (int): integer for the threshold to binarization
        Returns:
            binary_image: binarization of the exodus
        """
        _, tresh = cv2.threshold(img,np.percentile(img,threshold),255,cv2.THRESH_BINARY)
        #Dilate = cv2.dilate(tresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)
        Opening = cv2.morphologyEx(tresh,cv2.MORPH_OPEN,kernel,iterations=1)
        Closing = cv2.morphologyEx(Opening,cv2.MORPH_CLOSE,kernel,iterations=1)   
        return Closing
    
    def getHardExodus(self,thresholdList):
        """Get all the exodus of the multy flow
        Args:
            thresholdList (list): 3 element list to get the exodus
        Returns:
            lists: lists with the multiple results
        """
        with tqdm(total=self.data_length,desc="Hard Exodus Extraction "+self.dataname) as pbar:
            for i in range(0,self.data_length):
                
                img1 = self.hardExodus2(self.hardexodus[i],thresholdList[0])
                self.result.append(img1)
                ip=self.hardExodus(self.hardexodus[i],thresholdList[1],cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6)))
                img2 = self.hardExodus(self.hardexodus[i],thresholdList[2],cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
                self.result2.append(img2)
                self.result3.append(cv2.bitwise_or(img1,img2))
                pbar.update(1)
                
        return self.result, self.result2, self.result3
    