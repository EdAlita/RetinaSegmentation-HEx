import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern

class feature_extractor():
    def __init__(self):
        self.distances = [1, 2]
        self.angles = [0, np.pi / 4]  # Example angles
        self.properties = ['correlation', 'homogeneity', 'contrast', 'energy', 'dissimilarity']
        self.gray_image = None
        self.binary_image = None
        self.contours = None
        self.angles_Haralick = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        self.radius = 1
        self.n_points = 8 * self.radius #Each pixel's neighborhood defined by an 8-connected circle        
        
    def extract_features(self,image):
        features = []
        self.gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, self.binary_image = cv2.threshold(self.gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self.contours, _ = cv2.findContours(self.binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Texture Descriptors - GLCM, LBP, Haralick's
        glcm_features = self.extract_glcm_features()
        lbp_features = self.extract_lbp_features()
        haralick_features = self.extract_haralick_features()

        features.extend(glcm_features)
        features.extend(lbp_features)
        features.extend(haralick_features)

        # Shape Descriptors - Hu Moments, Area, Perimeter, Centroid coordinates
        hu_moments = self.extract_hu_moments()
        area = self.extract_area()
        perimeter = self.extract_perimeter()
        #centroid_coordinates = self.extract_centroid_coordinates()

        features.extend(hu_moments)
        features.append(area)
        features.append(perimeter)
        #features.extend(centroid_coordinates)

        # Color Descriptors - Mean value of the whole image and histogram mean value
        color_mean = self.extract_color_mean(image)
        histogram_mean = self.extract_histogram_mean()

        #features.append(color_mean)
        features.append(histogram_mean)

        return features

    def extract_glcm_features(self):
        # Extract GLCM features using gray-level co-occurrence matrix
        # Return a list of GLCM features
        glcms = []
        for distance in self.distances:
            for angle in self.angles:
                glcm = graycomatrix(self.gray_image, [distance], [angle], levels=256, symmetric=True, normed=True)
                glcms.append(glcm)

        features = []
        for i, glcm in enumerate(glcms):
            distance = self.distances[i // len(self.angles)]
            angle = self.angles[i % len(self.angles)]
            for prop in self.properties:
                feature = graycoprops(glcm, prop).ravel()
                features.extend(feature)

        return features

    def extract_lbp_features(self):
        # Extract LBP features using Local Binary Patterns
        # Return a list of LBP features
        features = []

        lbp = local_binary_pattern(self.gray_image, self.n_points, self.radius, method='uniform') #Mapping used-method uniform. Binary pattern (uniform patterns)
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, self.n_points + 3), range=(0, self.n_points + 2), density=True) #Calculates and returns the normalized histogram

        features.extend(hist.tolist())

        return features

    def extract_haralick_features(self):
        # Extract Haralick's texture features
        # Return a list of Haralick features
        features = []
        for distance in self.distances:
            for angle in self.angles_Haralick:
                glcm = graycomatrix(self.gray_image, [distance], [angle], levels=256, symmetric=True, normed=True)
                for prop in self.properties:
                    feature = graycoprops(glcm, prop).ravel()
                    features.extend(feature)

        return features

    def extract_hu_moments(self):
        #Extract Hu Moments as shape descriptors
        #Return a list of Hu Moments
        hu_moments = []

        #Calculate moments
        moments = cv2.moments(self.binary_image)
        hu_moments_raw = cv2.HuMoments(moments)
        hu_moments_normalized = -np.sign(hu_moments_raw) * np.log10(np.abs(hu_moments_raw))

        #Append Hu Moments to the list
        for hu_moment in hu_moments_normalized:
            hu_moments.append(hu_moment[0])

        return hu_moments

    def extract_area(self):

        area = 0
        if len(self.contours) > 0:
            largest_contour = max(self.contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

        return area

    def extract_perimeter(self):
        
        # Calculate the perimeter of the largest contour (assuming it as the region of interest)
        perimeter = 0
        if len(self.contours) > 0:
            largest_contour = max(self.contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)

        return perimeter

    def extract_centroid_coordinates(self):

        #Calculate the centroid coordinates of the largest contour (assuming it as the region of interest)
        centroid_coordinates = []
        if len(self.contours) > 0:
            largest_contour = max(self.contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroid_coordinates = [cX, cY]

        return centroid_coordinates

    def extract_color_mean(self,image):
        #Convert the image to the LAB color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        #Split the LAB image into L, A, and B channels
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        #Calculate the mean value of the A and B channels
        a_mean = np.mean(a_channel)
        b_mean = np.mean(b_channel)

        #Combine the mean values into a color descriptor
        color_mean = np.array([a_mean, b_mean])

        return color_mean

    def extract_histogram_mean(self):

        # Calculate the histogram of the grayscale image
        hist, _ = np.histogram(self.gray_image.flatten(), bins=256, range=[0, 256])

        # Calculate the mean value of the histogram
        histogram_mean = np.mean(hist)

        return histogram_mean