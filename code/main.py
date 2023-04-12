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
parser.add_argument("-ll","--lowerlimit", default="l",help='Gives the lower limit to cut the data. Default value is the entire array of Data')
parser.add_argument("-lh","--highlimit", default="h",help='Gives the higher limit to cut the data. Default value is the entire array of Data')

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

img = np.zeros((2848, 4288, 3), dtype = "uint8")

ds_tr = []
ds_ts = []

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
ds_ts_pp = prepos(timestr,trash,"Testing",ds_ts[args.lowerlimit:args.highlimit],intermedateResult=int(args.intermedateresults))
#ds_tr_pp = prepos(timestr,trash,"Trainning",ds_tr)
###Creating Mask
main_logger.debug("Preprocessing had finnish")

ds_ts_mask = masks(timestr,trash,"Testing",ds_ts[args.lowerlimit:args.highlimit],intermedateResult=int(args.intermedateresults))
main_logger.debug("Masking had finnish")

main_logger.debug("The code run was sucessful")
main_logger.debug("exit code 0")