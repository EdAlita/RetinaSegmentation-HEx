from glob import glob

#Open config file to get the ṕath of our images.

"""
Before running create a file name 

main.cfg 

with the next lines: 

Path to test images
Path to training images

.gitignore will not let this file upload to this file to our file structure

"""
with open('main.cfg', 'r') as file:
    test = file.readline().rstrip()
    training = file.readline().rstrip()

#Creating data sets of all the images.

ds_ts = [ cv2.imread(fn, cv2.IMREAD_COLOR) for fn in glob(test) ]
ds_tr = [ cv2.imread(fn, cv2.IMREAD_COLOR) for fn in glob(training) ]