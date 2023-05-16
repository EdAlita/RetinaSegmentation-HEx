import cv2
import time
import os
from proj_functions import *
from preposcessing import prepos
from masks import masks
from extendable_logger import *
import numpy as np
import argparse
from opticaldisk import opticaldisk
from hard_exodus import hardExodusSegmentation
import matplotlib.pyplot as plt
import pandas as pd

start_time = time.time()
#####Creating the local variables use in this project
#Empty Image for processing and empty lists
test_image = np.zeros((2848, 4288, 3), dtype = "uint8")
training_image = np.zeros((2848, 4288, 3),dtype="uint8")
hard_exodus = np.zeros((2848, 4288, 3),dtype="uint8")
training_dataset = []
test_dataset = []
training_groundthruth_dataset = []

###Parsing de arguments of the cmd line
parser = argparse.ArgumentParser("Project to detected Hard and Soft Exodus")
parser.add_argument("-l", "--level", default=0,help='level of the internal logger (default: 0 , will not create logs)For more help check wiki on loggers for the correct value.')
parser.add_argument("-ir","--intermedateresults", default=0,help='Creates intermedate results of the number that you provide.Store them in Log folder. Provide a number')
parser.add_argument("-ll","--lowerlimit", default=100,help='Gives the lower limit to cut the data. Default value is the entire array of Data')
parser.add_argument("-lh","--highlimit", default=100,help='Gives the higher limit to cut the data. Default value is the entire array of Data')

#Geting all the the args
arguments = parser.parse_args()
#gettingCurrentpath
currentpath = os.getcwd()
file_structure(currentpath)

#Allow Logging function
loglevel = int(arguments.level)
#Creating the time of running the code
timestamp = time.strftime("%m%d%Y-%H%M%S")

### Creating Log only if you pass the level in command line
main_logger = creatingLogStructure("main.log",loglevel,os.path.join(currentpath,'logs',timestamp),timestamp)
main_logger.debug("Begin of the main.py code")

#Open file to obtain local path to the data field
filename = 'main.cfg'
test_path,training_path,training_groundtruths_path = get_localDirectories(filename,main_logger)

#Creating data sets of all the images.
test_names= os.listdir(test_path)
training_names= os.listdir(training_path)
training_groundtruths_names=os.listdir(training_groundtruths_path)
#sorting the images
test_names.sort()
training_names.sort()
training_groundtruths_names.sort()

#Getting len of the data
testList_length = len(test_names)
trainingList_length = len(training_names)

testList_lowerlimit,trainingList_lowerlimit = settingLimits(arguments.lowerlimit,0,0)
testList_highlimit, trainingList_highlimit = settingLimits(arguments.highlimit,testList_length,trainingList_length)

#Reading all the images and append it to the empty list
for i in range(0,testList_length):
    test_image = cv2.imread(test_path+test_names[i],cv2.COLOR_BGR2RGB)
    test_dataset.append(test_image)
    
for i in range(0,trainingList_length):
    training_image = cv2.imread(training_path+training_names[i],cv2.COLOR_BGR2RGB) 
    training_groundtruth = cv2.imread(training_groundtruths_path+training_groundtruths_names[i],cv2.IMREAD_GRAYSCALE)
    training_dataset.append(training_image)
    training_groundthruth_dataset.append(training_groundtruth)

main_logger.debug("The list length of the test is "+str(len(test_dataset)))
main_logger.debug("The list length of the training is "+str(len(training_dataset)))

###Creating Masks
"""
main_logger.debug("Masking had beging")
test_masks = masks(timestamp,loglevel,"Testing",test_dataset[testList_lowerlimit:testList_highlimit],intermedateResult=int(arguments.intermedateresults))
training_masks = masks(timestamp,loglevel,"Trainning",training_dataset[trainingList_lowerlimit:trainingList_highlimit],intermedateResult=int(arguments.intermedateresults))
main_logger.debug("Masking had finnish")

directory_last = os.path.join(currentpath,'Results','Masks','Tests')
save_images(test_masks[testList_lowerlimit:testList_highlimit],test_names[testList_lowerlimit:testList_highlimit],"Testing",directory_last,main_logger,"Masks")
directory_last = os.path.join(currentpath,'Results','Masks','Training')
save_images(training_masks[trainingList_lowerlimit:trainingList_highlimit],training_names[trainingList_lowerlimit:trainingList_highlimit],"Training",directory_last,main_logger,"Masks")
"""
###Deleting the Optical Disk
"""
main_logger.debug("Optical Disk Removal had begging")

test_removeOpticalDisk = opticaldisk(timestamp,loglevel,"Testing",test_dataset[testList_lowerlimit:testList_highlimit])
training_removeOpticalDisk  = opticaldisk(timestamp,loglevel,"Training",training_dataset[trainingList_lowerlimit:trainingList_highlimit])

directory_last = os.path.join(currentpath,'Results','OpticalDisk','Tests')
save_images(test_removeOpticalDisk[testList_lowerlimit:testList_highlimit],test_names[testList_lowerlimit:testList_highlimit],"Testing",directory_last,main_logger,"OpticalDisk")

directory_last = os.path.join(currentpath,'Results','OpticalDisk','Training')
save_images(training_removeOpticalDisk[trainingList_lowerlimit:trainingList_highlimit],training_names[trainingList_lowerlimit:trainingList_highlimit],"Training",directory_last,main_logger,"OpticalDisk")

main_logger.debug("Optical Disk Removal had finnish")
"""

###Create Preposcessing
main_logger.debug("Prepocessing had begging")

test_greenchannel,test_denoising = prepos(timestamp,loglevel,"Testing",test_dataset[testList_lowerlimit:testList_highlimit],intermedateResult=int(arguments.intermedateresults))
training_greenchannel,training_denoising = prepos(timestamp,loglevel,"Trainning",training_dataset[trainingList_lowerlimit:trainingList_highlimit],intermedateResult=int(arguments.intermedateresults))

directory_last = os.path.join(currentpath,'Results','Prepos','Tests')
save_images(test_denoising[testList_lowerlimit:testList_highlimit],test_names[testList_lowerlimit:testList_highlimit],"Testing",directory_last,main_logger,"Prepos")

directory_last = os.path.join(currentpath,'Results','Prepos','Training')
save_images(training_denoising[trainingList_lowerlimit:trainingList_highlimit],training_names[trainingList_lowerlimit:trainingList_highlimit],"Training",directory_last,main_logger,"Prepos")

main_logger.debug("Preprocessing had finnish")


###Hard Exodus

main_logger.debug("Hard Exodus had beging")

test_hardExodus = hardExodusSegmentation(timestamp,loglevel,"Test",test_denoising[testList_lowerlimit:testList_highlimit])
training_hardExodus = hardExodusSegmentation(timestamp,loglevel,"Training",training_denoising[trainingList_lowerlimit:trainingList_highlimit])

directory_last = os.path.join(currentpath,'Results','HardExodus','Tests')
save_images(test_hardExodus[testList_lowerlimit:testList_highlimit],test_names[testList_lowerlimit:testList_highlimit],"Testing",directory_last,main_logger,"HardExodus")

directory_last = os.path.join(currentpath,'Results','HardExodus','Training')
save_images(training_hardExodus[trainingList_lowerlimit:trainingList_highlimit],training_names[trainingList_lowerlimit:trainingList_highlimit],"Traininging",directory_last,main_logger,"HardExodus")

Precisions = []
Recalls = []
Index = []

for i in range(0,trainingList_highlimit):
    #img_resized = cv2.resize(training_groundthruth_dataset[i],None,fx=0.60,fy=0.60)
    precision = precision_score_(training_groundtruth[i],training_hardExodus[i])
    recall = recall_score_(training_groundtruth[i],training_hardExodus[i])
    Precisions.append(precision)
    Recalls.append(recall)
    Index.append("IDRiD_0{}".format(i+1))
    print("IDRiD_0{}: Precision: {} | Recall: {}".format(i+1,precision,recall))
    
d = {'Precision': pd.Series(precision,index=Index),
     'Recall' : pd.Series(recall,index=Index)   
} 

main_logger.debug("Hard Exodus had ending")

def evaluate_exodus(exodus,groud_truth,original_image):
    contours, _ = cv2.findContours(exodus,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    idx=0
    exodus_features = pd.DataFrame()
    exodus_labels = []
    
    for cnt in contours:
        idx += 1
        x,y,w,h = cv2.boundingRect(cnt)
        regions_exodus = exodus[y:y+h,x:x+w]
        regions_groundtruth = groud_truth[y:y+h,x:x+w]
        area_evaluation = evaluation(regions_exodus,regions_groundtruth)
        exodus_features = exodus_features.append(cv2.cvtColor(original_image[y:y+h,x:x+h],cv2.COLOR_RGB2GRAY))
        if ( area_evaluation > 0.1):
            exodus_labels.append(1)
        else:
            exodus_labels.append(0)
    return exodus_features,exodus_labels

contours, _ = cv2.findContours(training_groundthruth_dataset[2],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
cv2.drawContours(training_hardExodus[1],contours,-1,(125,0,0),4)

cv2.imwrite("test.jpg",training_hardExodus[1])

        
main_logger.debug("The code run was sucessful")
main_logger.debug("exit code 0")

end_time = time.time()

elapsed_time = end_time - start_time

elapsed_time = elapsed_time/60

hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print("Program ended the elapsed time is {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

