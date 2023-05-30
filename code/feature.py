import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from loguru import logger

class feature():
    def __init__(self):
        self.distances = [1,3,5]
        self.angles = [0, np.pi/4, np.pi/2 , 3*np.pi/4]
        self.properties = ['correlation', 'homogeneity', 'contrast', 'energy', 'dissimilarity']
            
    def calculate_glcms(self,matrix):
        # Calculate GLCM with specified distances and angles
        glcms = []
        for distance in self.distances:
            for angle in self.angles:
                glcm = graycomatrix(matrix, [distance], [angle], levels=256, symmetric=True, normed=True)
                glcms.append(glcm)

        # Calculate desired GLCM properties
        features = []
        out = pd.Series()
        
        for i, glcm in enumerate(glcms):
            distance = self.distances[i // len(self.angles)]
            angle = self.angles[i % len(self.angles)]
            for prop in self.properties:
                feature = graycoprops(glcm, prop).ravel()
                features.append(feature)
        
        return np.ravel(features)



