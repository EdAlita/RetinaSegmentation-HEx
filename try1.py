# import library 

import os
import numpy as np
import pywt
import cv2
from skimage.feature import greycomatrix, greycoprops
import csv
import pandas as pd

# path setup
data_dir=os.path.expanduser('/Users/taiaburrahman/Desktop/git/RetinaSegmentation-HEx/Results/HardExodus/Tests/')
len(os.listdir(data_dir))

files = []
labels=[]
# r=root, d=directories, f = files
for r, d, f in os.walk(data_dir):
    for file in f:
        if '.jpg' in file:
            label=r.split('\\')[-1]
            labels.append(label)
            files.append(os.path.join(r,file))
            


with open('hardx_features.csv', "a+", newline="") as wr:
    writer = csv.writer(wr)
    i=0
    for f in files:
        label=f.split('\\')[-1]
        img=cv2.imread(f)
        coeffs2 = pywt.dwt2(img, 'db2')
        LL, (LH, HL, HH) = coeffs2
        g=[]
        con=[]
        enr=[]
        dis=[]
        hom=[]
        g.append(greycomatrix(np.uint8(LL[:,:,0]), [1], [0], levels=256,normed=True,symmetric=True))
        g.append(greycomatrix(np.uint8(LL[:,:,1]), [1], [0], levels=256,normed=True,symmetric=True))
        g.append(greycomatrix(np.uint8(LL[:,:,2]), [1], [0], levels=256,normed=True,symmetric=True))
        g.append(greycomatrix(np.uint8(LH[:,:,0]), [1], [0], levels=256,normed=True,symmetric=True))
        g.append(greycomatrix(np.uint8(LH[:,:,1]), [1], [0], levels=256,normed=True,symmetric=True))
        g.append(greycomatrix(np.uint8(LH[:,:,2]), [1], [0], levels=256,normed=True,symmetric=True))
        g.append(greycomatrix(np.uint8(HL[:,:,0]), [1], [0], levels=256,normed=True,symmetric=True))
        g.append(greycomatrix(np.uint8(HL[:,:,1]), [1], [0], levels=256,normed=True,symmetric=True))
        g.append(greycomatrix(np.uint8(HL[:,:,2]), [1], [0], levels=256,normed=True,symmetric=True))
        g.append(greycomatrix(np.uint8(HH[:,:,0]), [1], [0], levels=256,normed=True,symmetric=True))
        g.append(greycomatrix(np.uint8(HH[:,:,1]), [1], [0], levels=256,normed=True,symmetric=True))
        g.append(greycomatrix(np.uint8(HH[:,:,2]), [1], [0], levels=256,normed=True,symmetric=True))
        for t in range(0,len(g)):
            con.append(greycoprops(np.array(g[t]), 'contrast'))
            enr.append(greycoprops(np.array(g[t]), 'energy'))
            dis.append(greycoprops(np.array(g[t]), 'dissimilarity'))
            hom.append(greycoprops(np.array(g[t]), 'homogeneity'))
        con_features=np.reshape(np.array(con).ravel(),(1,len(np.array(con).ravel())))
        enr_features=np.reshape(np.array(enr).ravel(),(1,len(np.array(enr).ravel())))
        dis_features=np.reshape(np.array(dis).ravel(),(1,len(np.array(dis).ravel())))
        hom_features=np.reshape(np.array(hom).ravel(),(1,len(np.array(hom).ravel())))
        features=np.concatenate((con_features,enr_features,dis_features,hom_features),axis=1);
        ff=features[0].tolist()     
        writer.writerow(ff+[labels[i]])
        i+=1
    wr.close()

print("print db")
print(len(files))
features_db=pd.read_csv('hardx_features.csv',header=None)
features_db.tail()