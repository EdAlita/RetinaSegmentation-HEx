import cv2
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from feature import feature 
class proj_functions():
    
    def __init__(self):
        self.cfgfile_path = os.path.join(os.getcwd(),'main.cfg')
        self.subsubfolders = ['Tests','Training']
        self.subfolders = ['HardExodus_92','HardExodus_97','Prepos']
        self.currentpath = os.getcwd()
    
    def get_localDirectories (self):
        
        if not (os.path.exists(self.cfgfile_path)):
            testFolderLocation = os.path.join('..','data','images','test/')
            trainingFolderLocation = os.path.join('..','data','images','training/')
            trainingGroundTruths = os.path.join('..','data','groundtruths','training','hard exudates/')
        else:
            with open(self.cfgfile_path, 'r') as file:
                testFolderLocation = file.readline().rstrip()
                trainingFolderLocation = file.readline().rstrip()
                trainingGroundTruths = file.readline().rstrip()
        return testFolderLocation,trainingFolderLocation,trainingGroundTruths
    
    def save_images(self,imageList,nameList,folderName,directory,process):
        numberofImages = len(imageList)
        with tqdm(total=numberofImages,desc="Saving Images "+folderName+" of "+process) as statusbar:
            for i in range(0,numberofImages):
                cv2.imwrite(os.path.join(directory,nameList[i]),imageList[i])
                statusbar.update(1)
    
    def settingLimits(self,argumentsLimit,firstdefaultvalue,seconddefaultvalue):
        if(argumentsLimit==100):
            firstlimit=firstdefaultvalue
            secondlimit=seconddefaultvalue
        else:
            firstlimit=int(argumentsLimit)
            secondlimit=int(argumentsLimit)
        return firstlimit, secondlimit
    
    def TestTrainnig(self,above_path):
        for element in self.subsubfolders:
            path = os.path.join(above_path,element)
            if not os.path.exists(path):
                os.mkdir(path)
    
    def SubFolders(self,above_path):
        for element in self.subfolders:
            path = os.path.join(above_path,element)
            if not os.path.exists(path):
                os.mkdir(path)
                self.TestTrainnig(path)
                
    def file_structure(self):
        Result = os.path.join(self.currentpath,'Results')
        if not (os.path.exists(Result)):
            os.mkdir(Result)
            self.SubFolders(Result)
    
    def evaluation(self,image,ground_truth):
        image_zero_bits, image_one_bits, ground_truth_zero_bits, ground_truth_one_bits = [],[],[],[]
        for i in range(0,image.shape[0]):
            for j in range(0,image.shape[1]):
                bit_value_ground_truth = ground_truth[i][j]
                bit_value_image = image[i][j]
                if bit_value_ground_truth == 0:
                    ground_truth_zero_bits.append((i,j))
                else:
                    ground_truth_one_bits.append((i,j))
                if bit_value_image == 0:
                    image_zero_bits.append((i,j))
                else:
                    image_one_bits.append((i,j))
        true_positive = len(set(image_one_bits).intersection(set(ground_truth_one_bits)))
        false_negative = len(set(image_zero_bits).intersection(set(ground_truth_zero_bits)))
        result=0
        if(true_positive+false_negative) !=0:
            result = true_positive/(true_positive+false_negative)
        return result
    
    def calc_Sensitivity_Sets(self,truth, pred):
        np.seterr(all="ignore")
        truth = np.divide(truth,255)
        #print(np.max(truth))
        inv_pred = np.divide((255 - pred),255)
        pred = np.divide(pred, 255)
        TP = np.sum(pred*truth)
        FP  = np.sum(inv_pred*truth)
        sens = TP/(TP+FP)
        return sens
    
    def evaluate_exodus(self,exodus,groud_truth,original_image):
        count = 0 
        contours, _ = cv2.findContours(exodus,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        idx=0
        sensivities_out = []
        negative_exodus = []
        positive_exodus = []
        
        num_total_exodus = len(contours)
        
        
        feature_extraction = feature(
            [1,3,5],
            [0, np.pi/4, np.pi/2 , 3*np.pi/4],
            ['correlation', 'homogeneity', 'contrast', 'energy', 'dissimilarity'])
        
        count = 0
        
        for cnt in contours:
            idx += 1
            x,y,w,h = cv2.boundingRect(cnt)
            regions_exodus = exodus[y:y+h,x:x+w]
            regions_groundtruth = groud_truth[y:y+h,x:x+w]
            
            sensivities = self.calc_Sensitivity_Sets(regions_groundtruth,regions_exodus)
            area_evaluation = self.evaluation(regions_exodus,regions_groundtruth)
            
            if ( area_evaluation < 0.1):
                negative_exodus.append(feature_extraction.calculate_glcms(cv2.cvtColor(original_image[y:y+h,x:x+h],cv2.COLOR_BGR2GRAY)))
                
                
            if ( area_evaluation > 0.3):
                positive_exodus.append(feature_extraction.calculate_glcms(cv2.cvtColor(original_image[y:y+h,x:x+h],cv2.COLOR_BGR2GRAY)))
                sensivities_out.append(sensivities)
                count = count + 1
                
            
            if ( len(sensivities_out) == 0 ):
                sens = 0.0
            else: 
                sens = sum(sensivities_out)/len(sensivities_out)

              
        return negative_exodus, positive_exodus , sens , self.calc_Sensitivity_Sets(groud_truth,exodus), num_total_exodus, count
            
                
            
    
    
    
    