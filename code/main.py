import cv2
import time
import os
from proj_functions import *
from preposcessing import prepos
from masks import masks
from extendable_logger import extendable_logger
import numpy as np
import argparse

###Parsing 
parser = argparse.ArgumentParser("Project to detected Hard and Soft Exodus")
parser.add_argument("-l", "--level", default=0,help='level of the internal logger (default: 0 , will not create logs)For more help check wiki on loggers for the correct value.')
parser.add_argument("-ir","--intermedateresults", default=0,help='Creates intermedate results of the number that you provide.Store them in Log folder. Provide a number')
parser.add_argument("-ll","--lowerlimit", default=100,help='Gives the lower limit to cut the data. Default value is the entire array of Data')
parser.add_argument("-lh","--highlimit", default=100,help='Gives the higher limit to cut the data. Default value is the entire array of Data')



args = parser.parse_args()




#Allow Logging function
trash = int(args.level)
timestr = time.strftime("%m%d%Y-%H%M%S")

### Creating Log only if you pass the level in command line
if (trash!=0):
    sfname = "main.log"
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory,'logs',timestr) 
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    main_logger = extendable_logger('main',"logs/"+timestr+"/"+sfname,level=trash)
else:
    main_logger = extendable_logger('main',"tmp",trash)
    main_logger.disabled = True
    os.remove("tmp")
      
main_logger.debug("Begin of the main.py code")

"""
For Logging use this functions

logging.debug("Debug logging test...")
logging.info("Program is working as expected")
logging.warning("Warning, the program may not function properly")
logging.error("The program encountered an error")
logging.critical("The program crashed")
"""

"""
Before running create a file name 

main.cfg 

with the next lines: 

Path to test images
Path to training images

.gitignore will not let this file upload to this file to our file structure

"""

#Open file to obtain local variables
fname = 'main.cfg'
test,training = get_localDirectories(fname,main_logger)

#Creating data sets of all the images.

# Make empty list.
test_n= os.listdir(test)
training_n= os.listdir(training)
test_n.sort()
training_n.sort()

ll_ts = int(args.lowerlimit)
ll_tr = int(args.lowerlimit)

hl_ts = int(args.highlimit)
hl_tr = int(args.highlimit)

if(args.lowerlimit==100):
    ll_ts = 0
    ll_tr = 0
    
if(args.highlimit==100):
    hl_ts = len(test_n)
    hl_tr = len(training_n)

img = np.zeros((2848, 4288, 3), dtype = "uint8")

ds_tr = []
ds_ts = []

gc = []
gd = []

for i in range(0,len(test_n)):

    img = cv2.imread(test+test_n[i],cv2.IMREAD_COLOR)
    ds_ts.append(img)

for i in range(0,len(training_n)):
    img = cv2.imread(training+training_n[i],cv2.IMREAD_COLOR) 
    ds_tr.append(img) 

main_logger.debug("The list length of the test is "+str(len(ds_ts)))
main_logger.debug("The list length of the training is "+str(len(ds_tr)))

#cv2.imshow('sample image',ds_ts[4])
#Convert to gray scale.   
#ds_ts_gs = RGB2Gray( ds_ts,"Testing")
#ds_tr_gs = RGB2Gray( ds_tr,"Training")

# Now let's create a mask for this image

###Create Preposcessing
ds_ts_pp,gc,gd = prepos(timestr,trash,"Testing",ds_ts[ll_ts:hl_ts],intermedateResult=int(args.intermedateresults))
#ds_tr_pp = prepos(timestr,trash,"Trainning",ds_tr)
###Creating Mask
main_logger.debug("Preprocessing had finnish")



ds_ts_mask = masks(timestr,trash,"Testing",ds_ts[ll_ts:hl_ts],intermedateResult=int(args.intermedateresults))
#main_logger.debug("Masking had finnish")

cv2.imwrite('green_clahe.jpg',gc[0])
cv2.imwrite('greenoising_den.jpg',gd[0])

#Check this function
def vein_extraction(img,mask):
    image = np.zeros((2848, 4288), dtype = "uint8")
    marker = np.zeros((2848, 4288), dtype = "uint8")
    blur = np.zeros((2848, 4288), dtype = "uint8")
    veins= np.zeros((2848, 4288), dtype = "uint8")
    
    inv_img = 255 - img
    Inhance = adjust_gamma(inv_img, gamma=1.0)
    smooth = cv2.GaussianBlur(Inhance, (7, 7), 0)   
    #masked = cv2.bitwise_and(img,mask)
    clahe = cv2.createCLAHE(clipLimit=0.20, tileGridSize=(1,1))
    img_clahe = clahe.apply(smooth)
    stuctElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    marker = cv2.morphologyEx(smooth, cv2.MORPH_OPEN, stuctElement)
    final = cv2.subtract(marker,img_clahe)
    
    ret,thresh = cv2.threshold(final,0,256,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    ret2,thresh2 = cv2.threshold(thresh,ret,ret,cv2.THRESH_BINARY)
    masked = cv2.bitwise_and(thresh2,thresh2, mask=mask)
    out = cv2.medianBlur(masked,7)
    kernel= np.ones((5,5), np.uint8)
    out = cv2.dilate(out,kernel)
    

    contours, hierarchy = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    """
    for i, cnt in enumerate(contours):
        if hierarchy[0][i][0] != -1:
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri,True)
            if not len(approx) == 4:
                veins = cv2.drawContours(veins,[cnt],0,(255,255,255),-1)
    """
    veins = cv2.drawContours(veins,contours,-1,(255,255,255),-1)
    #veins = 255 - veins
    kernel= np.ones((5,5), np.uint8)
    out = cv2.dilate(veins,kernel)
    smooth = cv2.erode(out,kernel)
    #result = cv2.subtract(img,smooth)
    result = cv2.bitwise_and(smooth,img)
    return result

blur = cv2.GaussianBlur(gd[0], (7, 7), 0)
veins = vein_extraction(blur,ds_ts_mask[0])
cv2.imwrite('vein_extraction.jpg',veins)

main_logger.debug("The code run was sucessful")
main_logger.debug("exit code 0")