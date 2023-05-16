import cv2
import os
import tqdm

def fContour():
    currentpath = os.getcwd()
    list = os.path.join(currentpath,'data','groundtruths','training','hard exudates')+"/"
    dataname=os.listdir(list)

    
    for i in range(0, len(dataname)):
        # print(list+dataname[i])
        img_mask= cv2.imread(list+dataname[i],cv2.IMREAD_GRAYSCALE)
        # cv2.imshow("",img_mask)
        # # cv2.imwrite("mask.png",img_mask)
        # # mask = cv2.imread("mask.png");
        # mask = cv2.cvtColor(img_mask,cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(img_mask, 0, 200)
        # #display= display_image("L, L after clahe",[edged],"horizontal", 1/5)
        contours, hierarchy= cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_mask=cv2.cvtColor(img_mask,cv2.COLOR_GRAY2BGR)
        cimg=cv2.drawContours(img_mask, contours, -1, (0,255,0), 2)

        path = os.path.join(currentpath,'Results','Contour','Training')+"/"
        cv2.imwrite(path+dataname[i],cimg)


# img_mask= cv2.imread(image_mask,cv2.IMREAD_GRAYSCALE)
# cv2.imwrite("mask.png",img_mask)
# mask = cv2.imread("mask.png");
# mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
# edged = cv2.Canny(mask, 0, 200)
# #display= display_image("L, L after clahe",[edged],"horizontal", 1/5)
# contours, hierarchy= cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(len(contours))
# print(contours[1])
# cv2.drawContours(img, contours, -1, (0,0,255), 2)