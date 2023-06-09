import cv2
from alive_progress import alive_bar
import time
import os
import numpy as np
from feature import feature
from features_extra import feature_extractor
from loguru import logger
class proj_functions():

    
    def __init__(self):
        
        self.cfgfile_path = os.path.join(os.getcwd(),'main.cfg')
        self.subsubfolders = ['Tests','Training']
        self.subfolders = ['HardExodus_92','HardExodus_97','Prepos']
        self.currentpath = os.getcwd()
        logger.info(f"Class Initialized: {self.__class__}")
    
    def get_localDirectories (self):
        
        if not (os.path.exists(self.cfgfile_path)):
            testFolderLocation = os.path.join('..','data','images','test/')
            trainingFolderLocation = os.path.join('..','data','images','training/')
            trainingGroundTruths = os.path.join('..','data','groundtruths','training','hard exudates/')
            testGroundTruths = os.path.join('..','data','groundtruths','test','hard exudates/')

        else:
            with open(self.cfgfile_path, 'r') as file:
                testFolderLocation = file.readline().rstrip()
                trainingFolderLocation = file.readline().rstrip()
                trainingGroundTruths = file.readline().rstrip()
                testGroundTruths = file.readline().rstrip()
        return testFolderLocation,trainingFolderLocation,trainingGroundTruths,testGroundTruths
    
    def save_images(self,imageList,nameList,folderName,directory,process):
        numberofImages = len(imageList)
        with alive_bar(total=numberofImages,title="Saving Images "+folderName+" of "+process) as statusbar:
            for i in range(0,numberofImages):
                cv2.imwrite(os.path.join(directory,nameList[i]),imageList[i])
                time.sleep(0.01)
                statusbar()
    
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
    
    def calc_Sensitivity_Sets(self,truth,pred):
        true_positive = np.sum(truth)
        total_groundthruth  = np.sum(truth)
        sens = true_positive/total_groundthruth
        #print('sens {} true_positives {} total {}'.format(sens,true_positive,total_groundthruth))
        return sens
    
    
    def evaluate_exodus(self,exodus,groud_truth,original_image,num):
        
        count = 0 
        contours, _ = cv2.findContours(exodus,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        ground, _ = cv2.findContours(groud_truth,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        
        idx=0

        sensivities_out = []
        negative_exodus = []
        positive_exodus = []
        y_output_negative = []
        y_output_positive = []
        groundThruth = 0         
        count = 0
        
        feature_extraction = feature()
        groundThruth += len(ground)

        
        for cnt in contours:

            idx+=1
            img = np.zeros_like(exodus)
            exodus = cv2.drawContours(img, [cnt], 0, (255,255,255), -1)
            #regions_intersection = cv2.bitwise_and(exodus,groud_truth)
            
            #area_evaluation = self.calc_Sensitivity_Sets(regions_intersection,regions_union)
            x,y,w,h = cv2.boundingRect(cnt)
            for grnd in ground:
                im2 = np.zeros_like(exodus)
                groundTh = cv2.drawContours(im2, [grnd], 0, (255,255,255), -1)
                if idx == 2:
                    cv2.imwrite('ground.jpg',groundTh)
                regions_intersection = cv2.bitwise_and(exodus,groundTh)
                intersection = np.sum(regions_intersection)
                if(intersection!=0):
                    regions_union = cv2.bitwise_or(exodus,groundTh)
                    break
                else:
                    regions_union = cv2.bitwise_or(exodus,groundTh)
            
            area_evaluation = np.sum(regions_intersection)/np.sum(regions_union)
                   
            
            if ( area_evaluation >= 0.1):
                positive_exodus.append(feature_extraction.calculate_glcms(cv2.cvtColor(original_image[y:y+h,x:x+w], cv2.COLOR_RGB2GRAY)))
                sensivities_out.append(area_evaluation)
                y_output_positive.append([num,idx,1,area_evaluation])
                count = count + 1
            else:
                negative_exodus.append(feature_extraction.calculate_glcms(cv2.cvtColor(original_image[y:y+h,x:x+w], cv2.COLOR_RGB2GRAY)))
                y_output_negative.append([num,idx,0,area_evaluation])
                
                
            if ( len(sensivities_out) == 0 ):
                sens = 0.0
            else: 
                sens = sum(sensivities_out)/len(sensivities_out)
            
            sens = 0
              
        return negative_exodus, positive_exodus ,sens,groundThruth, count, y_output_negative, y_output_positive      
            
                
            
    
    
    
    