import cv2
from extendable_logger import projloggger
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np

def prepos(timestr,trash,dname,data,intermedateResult=0):
    """This a preprocessing function of the projects

    Args:
        timestr (str): Day and time to create the logs of the project
        trash (int): Level of debuging for the logs
        dname (str): Name of the data for the logs
        data (img_array): Array of data to apply the preprocessing 
        intermedateResult (int, optional): The number of intermedate result to save in pdf format of the results. Defaults to 0.

    Returns:
        img_array : prepos images with the alterations apply
    """
    
    img = np.zeros((2848, 4288, 3), dtype = "uint8")
    image = np.zeros((2848, 4288, 3), dtype = "uint8")
    bus = []
    cli= []
    green_ch_c = []
    green_ch_d = []
    
    pre_logger = projloggger('preposcessing',timestr,dname,trash,'tmp2')
    pre_logger.debug("Begin of the prepos.py code")
    
    ### Steps
    ### Gren channel splitting
    ### CLAHE
    ### Denoissing with non Locals
    ### Inverting Image
    
    ### green channel splitting and CLAHE
    pre_logger.debug("Begin of the Preproscessing of "+dname)
    with tqdm(total=len(data),desc="Preposcessing "+dname) as pbar:
        for i in range(0,len(data)):
            img = data[i]
            (B, G, R) = cv2.split(img)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            B = clahe.apply(B)
            G = clahe.apply(G)
            R = clahe.apply(R)
            green_ch_c.append(G)
            image_merge = cv2.merge([B, G, R ])
            cli.append(image_merge)
            pre_logger.info(dname+" image "+str(i)+" Preposcessing")
            pbar.update(1)
    pre_logger.debug("End of the Preproscessing of "+dname)
    
    #Denoising the Images
    pre_logger.debug("Begin of the Denoising of "+dname)
    with tqdm(total=len(cli),desc="Denoising of "+dname) as pbar:
        for i in range(0,len(cli)):
            img = cv2.fastNlMeansDenoisingColored(cli[i],None,3,3,21,7)
            img = abs(img - 255)
            (B, G, R) = cv2.split(img)
            green_ch_d.append(G )
            bus.append(img)
            pre_logger.info(dname+" image "+str(i)+" Denoising")
            pbar.update(1)
    pre_logger.debug("End of the Denoising of "+dname)
    
    #Printing intermedated Results
    if(intermedateResult!=0):
        pre_logger.debug("Creating intermedated results")
        plt.subplot(131),plt.imshow(cv2.cvtColor(data[intermedateResult], cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.subplot(132),plt.imshow(cv2.cvtColor(cli[intermedateResult], cv2.COLOR_BGR2RGB))
        plt.title("CLAHE")
        plt.subplot(133),plt.imshow(cv2.cvtColor(bus[intermedateResult], cv2.COLOR_BGR2RGB))
        plt.title("Denoising")
        plt.savefig("logs/"+timestr+"/PreposcessingResults"+str(intermedateResult)+".pdf")

    pre_logger.debug("The code run was sucessful")
    pre_logger.debug("exit code 0")
    
    prepos_data = bus
    return prepos_data, green_ch_c, green_ch_d 