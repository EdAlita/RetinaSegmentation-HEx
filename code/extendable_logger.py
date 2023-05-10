import logging
import os   

def extendable_logger(log_name, file_name,level):
    """Creates a logger with a extable functions

    Args:
        log_name (str): name to save the logger and call it later
        file_name (str): name of the file to save the information of the logger
        level (int): leve of the logging

    Returns:
        logger: a logger object with the intialize function given
    """
    handler = logging.FileHandler(file_name)
    handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s %(message)s'))
    specified_logger = logging.getLogger(log_name)
    specified_logger.setLevel(level)
    specified_logger.addHandler(handler)
    return specified_logger

def projloggger(function_name,timestr,dname,trash,tmp):
    """Function to create each logger from project info

    Args:
        function_name (str): Name of the function where the logger runs
        timestr (str): Date and time for saving the logger
        dname (str): name of the data analsysing
        trash (int): level of the logging
        tmp (): temporal name for creating and erasing the logger.

    Returns:
        logger: project logger for the function log
    """
    if (trash!=0):
       return extendable_logger(function_name,"logs/"+timestr+"/"+dname+function_name+".log",level=trash)
    else:
        #Create a different tmp number just for deleting so number no matters just not repeting
        logger = extendable_logger('main',tmp,trash)
        logger.disabled = True
        os.chmod(tmp, 0o777)
        os.remove(tmp)
        return logger
    
def creatingLogStructure(logname,loglevel,final_directory,timestamp):
    """Creates Initial logger Structure

    Args:
        logname (str): First logger name
        loglevel (int): level of logging for the first logger
        final_directory (path): os path to the final directory to store the loggers
        timestamp (str): Date and time for saving the logger

    Returns:
        logger: created logger for using
    """
    if (loglevel!=0):
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)
        logger = extendable_logger('main',"logs/"+timestamp+"/"+logname,level=loglevel)
    else:
        logger = extendable_logger('main',"tmp",loglevel)
        logger.disabled = True
        os.remove("tmp")
    return logger