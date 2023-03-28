from glob import glob
import cv2
import logging
import time

timestr = time.strftime("%Y%m%d-%H%M%S")

logging.basicConfig(filename=timestr+".log", level=logging.DEBUG)

logging.debug("Debug logging test...")
"""
For Logging use this functions

logging.debug("Debug logging test...")
logging.info("Program is working as expected")
logging.warning("Warning, the program may not function properly")
logging.error("The program encountered an error")
logging.critical("The program crashed")
"""

#Open config file to get the ṕath of our images.



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
with open(fname, 'r') as file:
    test = file.readline().rstrip()
    training = file.readline().rstrip()
logging.debug("Program open the cfg file")



#Creating data sets of all the images.

# Make empty list.

ds_ts = [ cv2.imread(fn, cv2.IMREAD_COLOR) for fn in glob(test) ]
ds_tr = [ cv2.imread(fn, cv2.IMREAD_COLOR) for fn in glob(training) ]

logging.debug("The list length of the test is "+str(len(ds_ts)))
logging.debug("The list length of the training is "+str(len(ds_tr)))


#cv2.imshow('sample image',ds_ts[4])

#Convert to gray scale.
for i in range(0,len(ds_ts)-1):
    ds_ts[i] = cv2.cvtColor(ds_ts[i], cv2.COLOR_BGR2GRAY)
    logging.info("Test image "+str(i)+" converted to grayscale")

for i in range(0,len(ds_tr)-1):
    ds_tr[i] = cv2.cvtColor(ds_tr[i], cv2.COLOR_BGR2GRAY)
    logging.info("Trainning image "+str(i)+" converted to grayscale")

logging.debug("Grayscale covertion finish without problems")


#cv2.imshow('grayscale',ds_ts[4])

#cv2.waitKey()