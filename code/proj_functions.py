import cv2
from tqdm import tqdm
import os

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
    folders = ['Tests','Trainning']
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