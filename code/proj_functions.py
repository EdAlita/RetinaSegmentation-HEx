import cv2
from tqdm import tqdm
import os
import numpy as np

def get_localDirectories ( cfgFile , logger):
    """Function to obtain the local Directories of the given files
    
    Args:
        cfgFile (str): name of the file to open
        logger (logger): logger to create a history of this function

    Returns:
        str: the two directions of the local files to use
    """
    cfgfile = os.path.join(os.getcwd(),'main.cfg')
    if not (os.path.exists(cfgfile)):
            testFolderLocation = os.path.join('..','data','images','test/')
            trainingFolderLocation = os.path.join('..','data','images','training/')
            trainingGroundTruths = os.path.join('..','data','groundtruths','training','hard exudates/')
    else:    
        with open(cfgFile, 'r') as file:
            testFolderLocation = file.readline().rstrip()
            trainingFolderLocation = file.readline().rstrip()
            trainingGroundTruths = file.readline().rstrip()
            logger.debug("Program open the cfg file") 
              
    return testFolderLocation,trainingFolderLocation,trainingGroundTruths

def save_images(imageList,nameList,folderName,directory,logger,process):
    """Function to save all the images in a folder

    Args:
        imageList (List): List of images to save
        nameList (List): List of the names
        foldername (str): Name of the folder to save the images
        directory (str): Directory to save the files
        logger (logger): Logger to create a history of this fucntion
        process (str): Name of the process to save
        numberofImages (int): length of the list of images

    Returns:
        None: no return
    """
    
    numberofImages = len(imageList)
    with tqdm(total=numberofImages,desc="Saving Images "+folderName+" of "+process) as statusbar:
        for i in range(0,numberofImages):
            cv2.imwrite(os.path.join(directory,nameList[i]),imageList[i])
            logger.info(nameList[i]+" image "+str(i)+" saving images of Prepos "+folderName)
            statusbar.update(1)  
                         
    return None

def settingLimits(argumentsLimit,firstdefaultvalue,seconddefaultvalue):
    """Set the limits from a default value or a setvalue

    Args:
        argumentsLimit (argument.limit): limit value of the argument given
        firstdefaultvalue (int): value to set
        seconddefaultvalue (int): value to set

    Returns:
        int: limits value
    """
    if(argumentsLimit==100):
        firstlimit=firstdefaultvalue
        secondlimit=seconddefaultvalue
    else:
        firstlimit=int(argumentsLimit)
        secondlimit=int(argumentsLimit)
    return firstlimit, secondlimit

def TestTrainnig(above_path):
    folders = ['Tests','Training']
    for element in folders:
        path = os.path.join(above_path,element)
        if not os.path.exists(path):
            os.mkdir(path)            
def SubFolders(above_path):
    folders = ['HardExodus','HardExodusJacks','OpticalDisk','Prepos']
    for element in folders:
        path = os.path.join(above_path,element)
        if not os.path.exists(path):
            os.mkdir(path)
        TestTrainnig(path)    
def file_structure(currentpath):
    Result = os.path.join(currentpath,'Results')
    if not (os.path.exists(Result)):
        os.mkdir(Result)
    SubFolders(Result)
    
def evaluation(image,ground_truth):
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

def precision_score_(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    total_pixel_pred = np.sum(pred_mask)
    precision = np.mean(intersect/total_pixel_pred)
    return round(precision, 3)

def recall_score_(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    total_pixel_truth = np.sum(groundtruth_mask)
    recall = np.mean(intersect/total_pixel_truth)
    return round(recall, 3)

def accuracy(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
    xor = np.sum(groundtruth_mask==pred_mask)
    acc = np.mean(xor/(union + xor - intersect))
    return round(acc, 3)

def dice_coef(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    total_sum = np.sum(pred_mask) + np.sum(groundtruth_mask)
    dice = np.mean(2*intersect/total_sum)
    return round(dice, 3) #round up to 3 decimal places

def iou(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
    iou = np.mean(intersect/union)
    return round(iou, 3)

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
        if ( area_evaluation < 0.1):
            exodus_labels.append(1)
        else:
            exodus_labels.append(0)
            
    return exodus_features,exodus_labels
            
                
            
    
    
    
    
