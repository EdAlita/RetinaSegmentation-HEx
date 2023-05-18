import cv2
from extendable_logger import projloggger
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np

def prepos(timestamp,loglevel,dataname,data,intermedateResult=0):
    """This a preprocessing function of the projects

    Args:
        timestamp (str): Day and time to create the logs of the project
        loglevel (int): Level of debuging for the logs
        dataname (str): Name of the data for the logs
        data (List): Array of data to apply the preprocessing 
        intermedateResult (int, optional): The number of intermedate result to save in pdf format of the results. Defaults to 0.

    Returns:
        image_list : prepos images with the alterations apply
    """
     
    clahe_image = []
    clahe_result= []
    denoising_result= []
    denoising_image = []
    
    pre_logger = projloggger('preposcessing',timestamp,dataname,loglevel,'tmp2')
    pre_logger.debug("Begin of the prepos.py code")
    
    data_length = len(data)
    
    ### Steps
    ### Gren channel splitting
    ### CLAHE
    ### Denoissing with non Locals
    ### Inverting Image
    
    ### green channel splitting and CLAHE
    pre_logger.debug("Begin of the Preproscessing of "+dataname)
    with tqdm(total=data_length,desc="Preposcessing "+dataname) as statusbar:
        for i in range(0,data_length):
            imageholder = data[i]
            #LAB_STRE= cv2.normalize(imageholder,None,alpha=imageholder.min(), beta=300,norm_type=cv2.NORM_MINMAX)
            (R, G, B) = cv2.split(imageholder) 
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
            G = clahe.apply(G)
            clahe_result.append(G)
            pre_logger.info(dataname+" image "+str(i)+" Preposcessing")
            statusbar.update(1)
    pre_logger.debug("End of the Preproscessing of "+dataname)
    
    #Denoising the Images
    pre_logger.debug("Begin of the Denoising of "+dataname)
    with tqdm(total=data_length,desc="Denoising of "+dataname) as statusbar:
        for i in range(0,data_length):
            imageholder = cv2.fastNlMeansDenoising(clahe_result[i],3,21,7)
            denoising_result.append(imageholder)
            pre_logger.info(dataname+" image "+str(i)+" Denoising")
            statusbar.update(1)
    pre_logger.debug("End of the Denoising of "+dataname)
    
    #Printing intermedated Results
    if(intermedateResult!=0):
        pre_logger.debug("Creating intermedated results")
        plt.subplot(131),plt.imshow(cv2.cvtColor(data[intermedateResult], cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.subplot(132),plt.imshow(cv2.cvtColor(clahe_image[intermedateResult], cv2.COLOR_BGR2RGB))
        plt.title("CLAHE")
        plt.subplot(133),plt.imshow(cv2.cvtColor(denoising_image[intermedateResult], cv2.COLOR_BGR2RGB))
        plt.title("Denoising")
        plt.savefig("logs/"+timestamp+"/PreposcessingResults"+str(intermedateResult)+".pdf")

    pre_logger.debug("The code run was sucessful")
    pre_logger.debug("exit code 0")
    
    return clahe_result, denoising_result 
