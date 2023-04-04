import logging
import time
import sys
import os

"""
Log access level: 

Critical 50
Error 40
Warning 30
Info 20
Debug 10
Notset 0

For Logging use this functions

logging.debug("Debug logging test...")
logging.info("Program is working as expected")
logging.warning("Warning, the program may not function properly")
logging.error("The program encountered an error")
logging.critical("The program crashed")
"""

class log:
    def __init__(self) -> None:
        pass
    
    def gen_file(self,filename):
        print(filename)
        if (len(sys.argv)!=1):
            trash = int(sys.argv[1])
            timestr = time.strftime("%m%d%Y-%H%M%S")
            
            current_directory = os.getcwd()
            final_directory = os.path.join(current_directory,'logs',timestr)
            if not os.path.exists(final_directory):
                os.makedirs(final_directory)
            logging.basicConfig(
                filename="logs/"+timestr+"/"+filename,
                level=trash,
                format="%(asctime)s - %(levelname)s - %(message)s",
          
                )
            
        logging.debug("Begin of the "+filename+" code")
    
    def add_info(self,info):
        logging.info(info)
    
    def add_debug(self,msg):
        logging.debug(msg)

    def add_warning(self,msg):
        logging.warning(msg)
    
    def add_error(self,msg):
        logging.error(msg)
    
    def add_critical(self,msg):
        logging.critical(msg)