from glob import glob
import cv2
import logging
import time
import sys
import os

#Allow Logging function
trash = 0

if (len(sys.argv)!=1):
    trash = int(sys.argv[1])
    timestr = time.strftime("%m%d%Y-%H%M%S")
    sfname = "main.log"
    
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory,'logs',timestr)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    
    logging.basicConfig(filename="logs/"+timestr+"/"+sfname,level=trash,format="%(asctime)s - %(levelname)s - %(message)s")
    
logging.debug("Begin of the main.py code")

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
#Define of the local functions

def RGB2Gray( img_list , type_img ):
    for i in range(0,len(img_list)-1):
        img_list[i] = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY)
        logging.info(type_img+" image "+str(i)+" converted to grayscale")
    return img_list

def get_localDirectories ( name ):
    with open(name, 'r') as file:
        one = file.readline().rstrip()
        two = file.readline().rstrip()
        logging.debug("Program open the cfg file")
        
    return one,two

#Open file to obtain local variables
fname = 'main.cfg'

test,training = get_localDirectories(fname)


#Creating data sets of all the images.

# Make empty list.

ds_ts = [ cv2.imread(fn, cv2.IMREAD_COLOR) for fn in glob(test) ]
ds_tr = [ cv2.imread(fn, cv2.IMREAD_COLOR) for fn in glob(training) ]

logging.debug("The list length of the test is "+str(len(ds_ts)))
logging.debug("The list length of the training is "+str(len(ds_tr)))

#cv2.imshow('sample image',ds_ts[4])

#Convert to gray scale.
    
ds_ts_gs = RGB2Gray( ds_ts,"Testing")
ds_tr_gs = RGB2Gray( ds_tr,"Training")

logging.debug("Grayscale covertion finish without problems")


#cv2.imshow('grayscale',ds_ts_gs[4])

#cv2.waitKey()

logging.debug("The code run was sucessful")
logging.debug("exit code 0")