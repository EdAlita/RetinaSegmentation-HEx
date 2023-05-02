from glob import glob
import cv2
import logging
import time
import sys
import os
from logGenerator import log


class imagecvt:
    def __init__(self) -> None:
        self.logg=log()
        self.logg.gen_file('main.log')

    def RGB2Gray(self, img_list , type_img ):
        for i in range(0,len(img_list)-1):
            img_list[i] = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY)
            # logging.info(type_img+" image "+str(i)+" converted to grayscale")
            cv2.imwrite(img_list[i],img_list[i])
            self.logg.add_info(type_img+" image "+str(i)+" converted to grayscale")
        return img_list

    def get_localDirectories (self,name ):
        with open(os.getcwd()+"/code/"+name, 'r') as file:
            one = file.readline().rstrip()
            two = file.readline().rstrip()
            self.logg.add_debug("Program open the cfg file")
        return one,two

#Open file to obtain local variables

p1=imagecvt()
logg=log()
sname='main.cfg'
test,training = p1.get_localDirectories(sname)


#Creating data sets of all the images.

# Make empty list.

ds_ts = [ cv2.imread(fn, cv2.IMREAD_COLOR) for fn in glob(test) ]
ds_tr = [ cv2.imread(fn, cv2.IMREAD_COLOR) for fn in glob(training) ]

logg.add_debug("The list length of the test is "+str(len(ds_ts)))
logg.add_debug("The list length of the training is "+str(len(ds_tr)))

#cv2.imshow('sample image',ds_ts[4])

#Convert to gray scale.
    
ds_ts_gs = p1.RGB2Gray( ds_ts,"Testing")
ds_tr_gs = p1.RGB2Gray( ds_tr,"Training")

logging.debug("Grayscale covertion finish without problems")

#cv2.imshow('grayscale',ds_ts_gs[4])
#cv2.waitKey(10000)

logg.add_debug("The code run was sucessful")
logg.add_debug("exit code 0 %slog"% __file__)