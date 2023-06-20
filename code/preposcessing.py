import cv2
from tqdm import tqdm
from loguru import logger

class preprocessing():
    """Manipulate images for Exodus Segmentation
    """
    def __init__(self,
                 dataname,
                 data,
                 ):
        """Init of the prepocessing classs

        Args:
            dataname (str): Names of the data set
            data (list): List of images to analyze
        """
        self.data = data
        self.dataname = dataname
        self.clahe_result= []
        self.denoising_result= []
        self.data_length = len(data)
        logger.info(f"Class Initialized: {self.__class__}")
        
    def green_ch_splitting_and_clahe(self, img):
        """Splitting into green channel and Normalazing the image

        Args:
            img (image): 3 channel image in RGB

        Returns:
            g: 1 channel image in grayscale
        """
        (R, G, B) = cv2.split(img) 
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
        G = clahe.apply(G)
        return G
        
    def get_Prepocessing(self):
        """Do the complete process for the full dataset

        Returns:
            list: of Images applied the Clahe and Denosing
        """
        
        with tqdm(total=self.data_length,desc="Clahe "+self.dataname) as statusbar:
            for i in range(0,self.data_length):
                    
                self.clahe_result.append(
                    self.green_ch_splitting_and_clahe(self.data[i])
                )
                statusbar.update(1)
                    
        with tqdm(total=self.data_length,desc="Denosing Image "+self.dataname) as statusbar:
            for i in range(0,self.data_length): 
                    
                self.denoising_result.append(
                    cv2.fastNlMeansDenoising(self.clahe_result[i],15, 7, 21 )
                )
                statusbar.update(1)
                    
        return self.denoising_result