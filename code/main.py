from glob import glob
import cv2
import logging
import time
import sys
import os
import logging
import enlighten
import numpy as np

testpath='data/images/test/'
trainingpath='data/images/training/'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
# Setup progress bar
manager = enlighten.get_manager()
pbar = manager.counter(total=24, desc='Processing step:', unit='ticks')

class main:
    def __init__(self) -> None:
        logg=log()
        logg.add_debug("exit code 0 %slog" % __file__)

    def rgb2gray(self):
        for imsl in range(55,79):
            img=cv2.imread(testpath +"IDRiD_"+ str(imsl) +".jpg")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("result/gray/IDRiD_"+ str(imsl) +".jpg",gray)
            ret,thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            cv2.imwrite("result/thresh/IDRiD_"+ str(imsl) +".jpg",thresh)
            thresh = cv2.medianBlur(thresh, 5)
            cv2.imwrite("result/median/IDRiD_"+ str(imsl) +".jpg",thresh)

            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Draw the contours on the original image
            cv2.drawContours(gray, contours, -1, (0, 0, 255), 2)

            cv2.imwrite("result/contours/IDRiD_"+ str(imsl) +".jpg",gray)

            # # reshape the image into a 2D array
            # img_reshaped = img.reshape(-1, 3).astype(np.float32)

            # # perform k-means clustering
            # k = 5
            # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            # flags = cv2.KMEANS_PP_CENTERS
            # _, labels, centers = cv2.kmeans(img_reshaped, k, None, criteria, 5, flags)

            # # assign colors to pixels based on cluster centers
            # for i in range(img_reshaped.shape[0]):
            #     img_reshaped[i] = centers[labels[i]]

            # # convert the image back to its original shape and display it
            # img_reshaped = img_reshaped.reshape(img.shape)
            # img_reshaped = np.uint8(img_reshaped)

            # cv2.imwrite("result/kmens/IDRiD_"+ str(imsl) +".jpg",img_reshaped)

            logger.info("Processing step %s" % imsl)
            pbar.update()


m1=main()

m1.rgb2gray()

# (1) image preprocessing 
# (2) candidate extraction 
# (3) feature extraction 
# (4) classification 
# (5) post processing.