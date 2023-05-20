import cv2
from tqdm import tqdm
import numpy as np

class HardExodus():
    """Class to create the Hard Exodus Segmentation
    """
    
    def __init__(self,
                 dataname,
                 data,
                 mediandata
                 ):
        """Default initialization
        Args:
            dataname (string): Name of data to analyze
            data (List): List of images
            mediandata (List ): List of images
        """
        self.dataname = dataname
        self.hardexodus = data
        self.medianData = mediandata
        self.result = []
        self.result2 = []
        self.kernel = np.ones((2,2),np.uint8) #cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,2))
        self.zeros = np.zeros((2848,4288),dtype=np.uint8)
        self.kernel2 = np.ones((3, 3), np.uint8)
        self.data_length = len(data)
    
    def jackhardExodus(self,img):
        _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
        dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 3)
        _, sure_fg = cv2.threshold(dist_transform, 0.05*dist_transform.max(), 255, cv2.THRESH_BINARY)
        
        return np.uint8(sure_fg)
        
    def hardExodus(self,img):
        """First flow for obtaning the exodus

        Args:
            img (image): image with preprosesing

        Returns:
            binary_image: binarization of the exodus
        """
        _, tresh = cv2.threshold(img,np.percentile(img,90),255,cv2.THRESH_BINARY)
        Opening = cv2.morphologyEx(tresh,cv2.MORPH_OPEN,self.kernel,iterations=1)
        Closing = cv2.morphologyEx(Opening,cv2.MORPH_CLOSE,self.kernel,iterations=1)   
        return Closing
    
    def getHardExodus(self):
        """Get all the exodus of the multy flow

        Returns:
            lists: lists with the multiple results
        """
        with tqdm(total=self.data_length,desc="Hard Exodus Extraction "+self.dataname) as pbar:
            for i in range(0,self.data_length):
                self.result.append(self.hardExodus(self.hardexodus[i]))
                self.result2.append(self.jackhardExodus(self.medianData[i]))
                pbar.update(1)
        return self.result, self.result2
    