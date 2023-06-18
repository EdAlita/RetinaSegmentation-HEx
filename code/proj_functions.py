import cv2
from tqdm import tqdm
import time
import os
import numpy as np
from feature import feature
from loguru import logger

class proj_functions():
    """fuctions for the project
    """
    
    def __init__(self):
        """Initialization of variables
        """
        self.cfgfile_path = os.path.join(os.getcwd(),'main.cfg')
        self.subsubfolders = ['Test','Train']
        self.subfolders = ['HardExodus_85','HardExodus_90','HardExodus_95','Prepos']
        self.currentpath = os.getcwd()
        logger.info(f"Class Initialized: {self.__class__}")
    
    def get_localDirectories (self):
        """Returns the locations of the folder to get data sets

        Returns:
            dir: locatios to the folders
        """
        
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
        """Saves the images in the location folder

        Args:
            imageList (list): list of the images to save
            nameList (list): names of the images to save
            folderName (str): Name of the folder to save
            directory (dir): Direction of the folder
            process (str): Name of the process for status bar
        """
        numberofImages = len(imageList)
        with tqdm(total=numberofImages,desc="Saving Images "+folderName+" of "+process) as statusbar:
            for i in range(0,numberofImages):
                cv2.imwrite(os.path.join(directory,nameList[i]),imageList[i])
                statusbar.update(1)
    
    def settingLimits(self,argumentsLimit,firstdefaultvalue,seconddefaultvalue):
        """Set Limits for the list

        Args:
            argumentsLimit (int): number to select value
            firstdefaultvalue (int): value to set
            seconddefaultvalue (int): value to set

        Returns:
            int: liits to set
        """
        if(argumentsLimit==100):
            firstlimit=firstdefaultvalue
            secondlimit=seconddefaultvalue
        else:
            firstlimit=int(argumentsLimit)
            secondlimit=int(argumentsLimit)
        return firstlimit, secondlimit
    
    def TestTrainnig(self,above_path):
        """Creates the test and trainnig Folders

        Args:
            above_path (dir): above folder
        """
        for element in self.subsubfolders:
            path = os.path.join(above_path,element)
            if not os.path.exists(path):
                os.mkdir(path)
    
    def SubFolders(self,above_path):
        """Creates folder to save results

        Args:
            above_path (dir): above folder
        """
        for element in self.subfolders:
            path = os.path.join(above_path,element)
            if not os.path.exists(path):
                os.mkdir(path)
                self.TestTrainnig(path)
                
    def file_structure(self):
        """Creates the file structure
        """
        Result = os.path.join(self.currentpath,'Results')
        if not (os.path.exists(Result)):
            os.mkdir(Result)
            self.SubFolders(Result)
    
    def evaluate_exodus(self,exodus,groud_truth,original_image,num,th):
        """Evaluating the exodus candidates from a binary image and extract features

        Args:
            exodus (binary image): Exodus Segmentation
            groud_truth (binary mask): Ground thruth of Exodus
            original_image (img): Original image to extract features
            num (int): number of the image
            th (int): number to indentify the segementation

        Returns:
            list: reuslts list
        """
        
        #Get the countours from the binary images
        count = 0 
        contours, _ = cv2.findContours(exodus,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        ground, _ = cv2.findContours(groud_truth,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        
        #id's of the contours
        idx=0

        #Create empty lists
        sensivities_out = []
        negative_exodus = []
        positive_exodus = []
        y_output_negative = []
        y_output_positive = []
        
        #Variables to count
        groundThruth = 0         
        count = 0
        
        #create the feature class
        feature_extraction = feature()
        
        #assing number of contours
        groundThruth = len(ground)

        #Analyze every contour encounter in exodus binary image
        for cnt in contours:
            #Assing id of the contour and creating a empty image
            idx+=1
            img = np.zeros_like(exodus)
            
            #Draw the exodus a empty image
            exodus = cv2.drawContours(img, [cnt], 0, (255,255,255), -1)
            #Get bounding rectangle
            x,y,w,h = cv2.boundingRect(cnt)
            #Analyze the new image with all the countours inside of the groundthruths
            for grnd in ground:
                im2 = np.zeros_like(exodus)
                #Draw each one of the countours in one image
                groundTh = cv2.drawContours(im2, [grnd], 0, (255,255,255), -1)
                #Check the intersection of the Exodus and the ground
                regions_intersection = cv2.bitwise_and(exodus,groundTh)
                intersection = np.sum(regions_intersection)
                #check if there is a intersection, if not repeat the cycle
                if(intersection!=0):
                    #Assing the union
                    regions_union = cv2.bitwise_or(exodus,groundTh)
                    break
                else:
                    regions_union = cv2.bitwise_or(exodus,groundTh)
            
            #Evaluation of the IuO
            area_evaluation = np.sum(regions_intersection)/np.sum(regions_union)
                   
            #IuO > 0.2 means it is a exodus
            if ( area_evaluation >= 0.2):
                positive_exodus.append(feature_extraction.calculate_glcms(cv2.cvtColor(original_image[y:y+h,x:x+w], cv2.COLOR_RGB2GRAY)))
                sensivities_out.append(area_evaluation)
                y_output_positive.append([th,num,idx,1,area_evaluation])
                count = count + 1
            else:
                negative_exodus.append(feature_extraction.calculate_glcms(cv2.cvtColor(original_image[y:y+h,x:x+w], cv2.COLOR_RGB2GRAY)))
                y_output_negative.append([th,num,idx,0,area_evaluation])
            
            sens = 0
              
        return negative_exodus, positive_exodus ,sens,groundThruth, count, y_output_negative, y_output_positive      
            
                
            
    
    
    
    