from glob import glob
import cv2
import logging
import time
import sys
import os
import logging
import enlighten
import numpy as np

# path setup
testpath='data/images/test/'
trainingpath='data/images/training/'

test_dataset=[]
training_dataset=[]
test_dataset_opticaldisk=[]
training_dataset_opticaldisk=[]

# Setup progress bar
manager = enlighten.get_manager()
pbar = manager.counter(total=24, desc='Processing step:', unit='ticks')


def _imread():
    for imsl in range(55,79):
        img_name="IDRiD_"+ str(imsl) +".jpg"
        img=cv2.imread(testpath +img_name)
        test_dataset.append(img)

        disk_img=extractDisk(img)
        cv2.imwrite("result/opticalDisk/test/"+img_name,disk_img)
        test_dataset_opticaldisk.append(disk_img)
        pbar.update()
        
    # for imsl in range(1,55):
    #     img_name="IDRiD_"+ str(imsl) +".jpg"
    #     img=cv2.imread(trainingpath +img_name)
    #     training_dataset.append(img)

    #     disk_img=extractDisk(img)
    #     cv2.imwrite("result/opticalDisk/training/"+ img_name,disk_img)
    #     test_dataset_opticaldisk.append(disk_img)


def extractDisk(img):
        """create a dark disk on the optical disk of the image
        Args:
            img (img): Image of creating the mask
        Returns:
            img: mask of the image
        """
        (B, G, R) = cv2.split(img)
        imageBlur = cv2.GaussianBlur(R,(25,25),0)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(imageBlur)
        img = cv2.merge([B, G, R])
        cv2.circle(img,maxLoc, 300, (0,0,0), -1)        
        return img

def opticalDisk():
    for imsl in range(len(test_dataset)):
        img=extractDisk(test_dataset[imsl])
        print("result/opticalDisk/test/"+ str(test_dataset[imsl]))


    # for imsl in range(len(training_dataset)):
    #     img=extractDisk(training_dataset[imsl])
    #     cv2.imwrite("result/opticalDisk/training/"+ str(training_dataset[imsl]),img)
    #     training_dataset_opticaldisk.append(img)

# Image load 
_imread()


# (1) image preprocessing 
# opticalDisk()
# (2) candidate extraction 
# (3) feature extraction 
# (4) classification 
# (5) post processing.