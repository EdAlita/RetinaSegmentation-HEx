
# Importing the libraries
import cv2
from proj_functions import *
  
# Reading the image and converting into B/W
# image = cv2.imread('book.png')

imageGT=cv2.imread("/Users/taiaburrahman/Desktop/git/RetinaSegmentation-HEx/data/groundtruths/training/hard exudates/IDRiD_10_EX.tif")
image=cv2.imread("/Users/taiaburrahman/Desktop/git/RetinaSegmentation-HEx/Results/HardExodus/Training/IDRiD_10.jpg")

precision = precision_score_(imageGT,image)

print(precision)

cv2.imshow("gt",imageGT)
cv2.imshow("img",image)
cv2.waitKey()

# from drawContour import fContour
# import cv2
# import os

# fContour()
# currentpath = os.getcwd()
# list = os.path.join(currentpath,'data','groundtruths','training','hard exudates')+"/"

# list = os.path.join(currentpath,'Results','Contour','Training')+"/"
# hardList = os.path.join(currentpath,'Results','HardExodus','Training')+"/"
# dataname=os.listdir(list)

# img=cv2.imread(list+dataname[0])


# cv2.imshow("img",img)
# cv2.waitKey(10000)

