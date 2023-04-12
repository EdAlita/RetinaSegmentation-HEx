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