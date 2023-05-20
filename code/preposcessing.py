import cv2
from tqdm import tqdm

class preprocessing():
    def __init__(self,
                 dataname,
                 data,
                 ):
        self.data = data
        self.dataname = dataname
        self.clahe_result= []
        self.denoising_result= []
        self.median_result = []
        self.data_length = len(data)
        
    def green_ch_splitting_and_clahe(self, img):
        imageholder = cv2.resize(img,None,fx=0.60,fy=0.60)
        (R, G, B) = cv2.split(imageholder) 
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
        G = clahe.apply(G)
        return G
        
    def get_Prepocessing(self):
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
                     
                    self.median_result.append(
                        cv2.medianBlur(self.clahe_result[i],5)
                    )
                    statusbar.update(1)
                    
        return self.clahe_result, self.denoising_result, self.median_result