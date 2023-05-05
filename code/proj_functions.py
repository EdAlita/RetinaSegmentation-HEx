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
    else:    
        with open(cfgFile, 'r') as file:
            testFolderLocation = file.readline().rstrip()
            trainingFolderLocation = file.readline().rstrip()
            logger.debug("Program open the cfg file") 
              
    return testFolderLocation,trainingFolderLocation

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
    folders = ['HardExodus','Masks','OpticalDisk','Prepos']
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
    
def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """

    YIQ = np.ndarray(imgRGB.shape)

    YIQ[:, :, 0] = 0.299 * imgRGB[:, :, 0] + 0.587 * imgRGB[:, :, 1] + 0.114 * imgRGB[:, :, 2]
    YIQ[:, :, 1] = 0.59590059 * imgRGB[:, :, 0] + (-0.27455667) * imgRGB[:, :, 1] + (-0.32134392) * imgRGB[:, :, 2]
    YIQ[:, :, 2] = 0.21153661 * imgRGB[:, :, 0] + (-0.52273617) * imgRGB[:, :, 1] + 0.31119955 * imgRGB[:, :, 2]

    return YIQ   

def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.59590059, -0.27455667, -0.32134392],
                             [0.21153661, -0.52273617, 0.31119955]])
    rgb_from_yiq = np.linalg.inv(yiq_from_rgb)

    RGB = np.ndarray(imgYIQ.shape)
    RGB[:, :, 0] = 1.00000001 * imgYIQ[:, :, 0] + 0.95598634 * imgYIQ[:, :, 1] + 0.6208248 * imgYIQ[:, :, 2]
    RGB[:, :, 1] = 0.99999999 * imgYIQ[:, :, 0] + (-0.27201283) * imgYIQ[:, :, 1] + (-0.64720424) * imgYIQ[:, :, 2]
    RGB[:, :, 2] = 1.00000002 * imgYIQ[:, :, 0] + (-1.10674021) * imgYIQ[:, :, 1] + 1.70423049 * imgYIQ[:, :, 2]

    return RGB 