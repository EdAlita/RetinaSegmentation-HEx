import numpy as np
import cv2
def RGB2Gray( img_list , type_img, logger):
    """gets the RGB image and coverted into Gray Scale

    Args:
        img_list (img): Image array to convert
        type_img (str): Name of the data to convert 
        logger (logger): logger to create the logs of the function

    Returns:
        img_array: Gray array of the image provide
    """
    #Converts a OpenCV array of images to Grayscale
    with tqdm(total=len(img_list),desc="Grayscale of "+type_img) as pbar:
        for i in range(0,len(img_list)):
            img_list[i] = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY)
            logger.info(type_img+" image "+str(i)+" converted to grayscale")
            pbar.update(i)
    return img_list

def get_localDirectories ( name , logger ):
    """Function to obtain the local Directories of the given files

    Args:
        name (str): name of the file
        logger (logger): logger to create a history of this function

    Returns:
        str: the two directions of the local files to use
    """
    #gets the data from the direction of the cfg file
    with open(name, 'r') as file:
        one = file.readline().rstrip()
        two = file.readline().rstrip()
        logger.debug("Program open the cfg file")   
    return one,two

def adjust_gamma(image, gamma=1.0):
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def OTSU(img_gray):
    max_g = 0
    suitable_th = 0
    th_begin = 0
    th_end = 256
    for threshold in range(th_begin, th_end):
        bin_img = img_gray > threshold
        bin_img_inv = img_gray <= threshold
        fore_pix = np.sum(bin_img)
        back_pix = np.sum(bin_img_inv)
        if 0 == fore_pix:
            break
        if 0 == back_pix:
            continue
 
        w0 = float(fore_pix) / img_gray.size
        u0 = float(np.sum(img_gray * bin_img)) / fore_pix
        w1 = float(back_pix) / img_gray.size
        u1 = float(np.sum(img_gray * bin_img_inv)) / back_pix
        # intra-class variance
        g = w0 * w1 * (u0 - u1) * (u0 - u1)
        if g > max_g:
            max_g = g
            suitable_th = threshold
 
    return suitable_th