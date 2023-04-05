import logging
import os

def extendable_logger(log_name, file_name,level):
    handler = logging.FileHandler(file_name)
    handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s %(message)s'))
    specified_logger = logging.getLogger(log_name)
    specified_logger.setLevel(level)
    specified_logger.addHandler(handler)
    return specified_logger

def projloggger(function_name,timestr,dname,trash,tmp):
    if (trash!=0):
       return extendable_logger('function_name',"logs/"+timestr+"/"+dname+"function_name.log",level=trash)
    else:
        #Create a different tmp number just for deleting so number no matters just not repeting
        logger = extendable_logger('main',"tmp3",trash)
        logger.disabled = True
        os.remove("tmp3")
        return logger