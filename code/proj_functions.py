import cv2
from tqdm import tqdm
import os
import pandas as pd
class proj_functions():
    
    def __init__(self):
        self.cfgfile_path = os.path.join(os.getcwd(),'main.cfg')
        self.subsubfolders = ['Tests','Training']
        self.subfolders = ['HardExodus','HardExodusJacks','Prepos']
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
    
    def evaluate_exodus(self,exodus,groud_truth,original_image):
        count = 0 
        contours, _ = cv2.findContours(exodus,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        idx=0
        exodus_features = pd.DataFrame()
        exodus_labels = []
        
        for cnt in contours:
            idx += 1
            x,y,w,h = cv2.boundingRect(cnt)
            regions_exodus = exodus[y:y+h,x:x+w]
            regions_groundtruth = groud_truth[y:y+h,x:x+w]
            area_evaluation = self.evaluation(regions_exodus,regions_groundtruth)
            
            #exodus_features = exodus_features.append(cv2.cvtColor(original_image[y:y+h,x:x+h],cv2.COLOR_RGB2GRAY))
            
            if ( area_evaluation < 0.1):
                exodus_labels.append(0)
            
            if ( area_evaluation > 0.3):
                exodus_labels.append(1)
                count = count + 1
                
        return count,len(contours)
            
                
            
    
    
    
    