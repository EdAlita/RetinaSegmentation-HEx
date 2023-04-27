import cv2
import time
import os
from proj_functions import *
from preposcessing import prepos
from masks import masks
from extendable_logger import extendable_logger
import numpy as np
import argparse

###Parsing de args of the cmd line
parser = argparse.ArgumentParser("Project to detected Hard and Soft Exodus")
parser.add_argument("-l", "--level", default=0,help='level of the internal logger (default: 0 , will not create logs)For more help check wiki on loggers for the correct value.')
parser.add_argument("-ir","--intermedateresults", default=0,help='Creates intermedate results of the number that you provide.Store them in Log folder. Provide a number')
parser.add_argument("-ll","--lowerlimit", default=100,help='Gives the lower limit to cut the data. Default value is the entire array of Data')
parser.add_argument("-lh","--highlimit", default=100,help='Gives the higher limit to cut the data. Default value is the entire array of Data')

#Geting all the the args
args = parser.parse_args()

#Allow Logging function
trash = int(args.level)
#Creating the time of running the code
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

#Open file to obtain local path to the data field
fname = 'main.cfg'
test,training = get_localDirectories(fname,main_logger)

#Creating data sets of all the images.
test_n= os.listdir(test)
training_n= os.listdir(training)
#sorting the images
test_n.sort()
training_n.sort()

#getting the lower and higer limit of the data to process
ll_ts = int(args.lowerlimit)
ll_tr = int(args.lowerlimit)
hl_ts = int(args.highlimit)
hl_tr = int(args.highlimit)

#Setting default lower and higher limits
if(args.lowerlimit==100):
    ll_ts = 0
    ll_tr = 0   
if(args.highlimit==100):
    hl_ts = len(test_n)
    hl_tr = len(training_n)

#Empty Image for processing and empty lists
img = np.zeros((2848, 4288, 3), dtype = "uint8")
ds_tr = []
ds_ts = []
gc_tr = []
gd_tr = []
gc_ts = []
gd_ts = []

mask_ts = []
mask_tr = []

#Reading all the images and append it to the empty list
for i in range(0,len(test_n)):
    img = cv2.imread(test+test_n[i],cv2.IMREAD_COLOR)
    ds_ts.append(img)
    
for i in range(0,len(training_n)):
    img = cv2.imread(training+training_n[i],cv2.IMREAD_COLOR) 
    ds_tr.append(img) 

main_logger.debug("The list length of the test is "+str(len(ds_ts)))
main_logger.debug("The list length of the training is "+str(len(ds_tr)))

# Now let's create a mask for this image

###Create Preposcessing
ds_ts_pp,gc_ts,gd_ts = prepos(timestr,trash,"Testing",ds_ts[ll_ts:hl_ts],intermedateResult=int(args.intermedateresults))
ds_tr_pp,gc_tr,gd_tr = prepos(timestr,trash,"Trainning",ds_tr[ll_tr:hl_tr],intermedateResult=int(args.intermedateresults))

crt = os.getcwd()
directory_last = os.path.join(crt,'Results','Prepos','Tests')
save_images(gd_ts,test_n,"Testing",directory_last,main_logger,"Prepos")
directory_last = os.path.join(crt,'Results','Prepos','Training')
save_images(gd_tr,training_n,"Training",directory_last,main_logger,"Prepos")


###Creating Mask
main_logger.debug("Preprocessing had finnish")
mask_ts = masks(timestr,trash,"Testing",ds_ts[ll_ts:hl_ts],intermedateResult=int(args.intermedateresults))
mask_tr = masks(timestr,trash,"Trainning",ds_tr[ll_tr:hl_tr],intermedateResult=int(args.intermedateresults))
main_logger.debug("Masking had finnish")

crt = os.getcwd()
directory_last = os.path.join(crt,'Results','Masks','Tests')
save_images(mask_ts,test_n,"Testing",directory_last,main_logger,"Masks")
directory_last = os.path.join(crt,'Results','Masks','Training')
save_images(mask_tr,training_n,"Training",directory_last,main_logger,"Masks")


main_logger.debug("The code run was sucessful")
main_logger.debug("exit code 0")