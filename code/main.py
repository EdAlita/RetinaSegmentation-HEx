import cv2,time,os,argparse
from proj_functions import proj_functions
from preposcessing import preprocessing
from hard_exodus import HardExodus
import numpy as np

class pipeline():
    def __init__(self):
        self.start_time = time.time()
        self.test_image = np.zeros((2848, 4288, 3), dtype = "uint8")
        self.training_image = np.zeros((2848, 4288, 3),dtype="uint8")
        self.hard_exodus = np.zeros((2848, 4288, 3),dtype="uint8")
        self.training_dataset = []
        self.test_dataset = []
        self.training_groundthruth_dataset = []
        self.parser = argparse.ArgumentParser("Project to detected Hard and Soft Exodus")
        self.parser.add_argument("-ll","--lowerlimit", default=100,help='Gives the lower limit to cut the data. Default value is the entire array of Data')
        self.parser.add_argument("-lh","--highlimit", default=100,help='Gives the higher limit to cut the data. Default value is the entire array of Data')
        self.currentpath = os.getcwd()
        self.test_prepos_path = os.path.join(self.currentpath,'Results','Prepos','Tests')
        self.training_prepos_path = os.path.join(self.currentpath,'Results','Prepos','Training')
        self.test_exodus_path = os.path.join(self.currentpath,'Results','HardExodus','Tests')
        self.training_exodus_path = os.path.join(self.currentpath,'Results','HardExodus','Training')
        self.test_exodusJ_path = os.path.join(self.currentpath,'Results','HardExodusJacks','Tests')
        self.training_exodusJ_path = os.path.join(self.currentpath,'Results','HardExodusJacks','Training')
        self.helpers = proj_functions()
        
    def preprocessing_featureExtraction(self):
        ##Geting all the the args
        arguments = self.parser.parse_args()
        
        self.helpers.file_structure()
        
        #Open file to obtain local path to the data field
        test_path,training_path,training_groundtruths_path = self.helpers.get_localDirectories()
        
        #Creating data sets of all the images.
        test_names= os.listdir(test_path)
        training_names= os.listdir(training_path)
        training_groundtruths_names=os.listdir(training_groundtruths_path)
        
        #sorting the images
        test_names.sort()
        training_names.sort()
        training_groundtruths_names.sort()
        
        #Getting len of the data
        testList_length = len(test_names)
        trainingList_length = len(training_names)
        
        testList_lowerlimit,trainingList_lowerlimit = self.helpers.settingLimits(arguments.lowerlimit,0,0)
        testList_highlimit, trainingList_highlimit = self.helpers.settingLimits(arguments.highlimit,testList_length,trainingList_length)
        #Reading all the images and append it to the empty list
        for i in range(0,testList_length):
            
            self.test_dataset.append(
                cv2.imread(test_path+test_names[i],cv2.COLOR_BGR2RGB)
            )
        
        for i in range(0,trainingList_length):
            
            self.training_dataset.append(
                cv2.imread(training_path+training_names[i],cv2.COLOR_BGR2RGB)
            )
            
            self.training_groundthruth_dataset.append(
                cv2.imread(training_groundtruths_path+training_groundtruths_names[i],cv2.IMREAD_UNCHANGED)
            )
        
        test_prepos = preprocessing(
            "Tests",
            self.test_dataset[testList_lowerlimit:testList_highlimit]
            )
        
        test_greenchannel,test_denoising,test_median = test_prepos.get_Prepocessing()
        
        training_prepos = preprocessing(
            "Training",
            self.training_dataset[trainingList_lowerlimit:trainingList_highlimit]   
        )
        
        training_greenchannel,training_denoising, training_median = training_prepos.get_Prepocessing()
        
        self.helpers.save_images(
            test_denoising[testList_lowerlimit:testList_highlimit],
            test_names[testList_lowerlimit:testList_highlimit],
            "Tests",
            self.test_prepos_path,
            "Prepos")
        
        self.helpers.save_images(
            training_denoising[trainingList_lowerlimit:trainingList_highlimit],
            training_names[trainingList_lowerlimit:trainingList_highlimit],
            "Training",
            self.training_prepos_path,
            "Prepos")
        
        ###Hard Exodus
        test_Exodus = HardExodus(
            "Test",
            test_denoising[testList_lowerlimit:testList_highlimit],
            test_median[testList_lowerlimit:testList_highlimit])
        
        training_Exodus = HardExodus(
            "Training",
            training_denoising[trainingList_lowerlimit:trainingList_highlimit],
            training_median[trainingList_lowerlimit:trainingList_highlimit])
        
        test_hardExodus, test_hardExodusJ = test_Exodus.getHardExodus()
        training_hardExodus, training_hardExodusJ= training_Exodus.getHardExodus()
        
        self.helpers.save_images(test_hardExodus[testList_lowerlimit:testList_highlimit],
                            test_names[testList_lowerlimit:testList_highlimit],
                            "Tests",
                            self.test_exodus_path,
                            "HardExodus")
        
        self.helpers.save_images(training_hardExodus[trainingList_lowerlimit:trainingList_highlimit],
                            training_names[trainingList_lowerlimit:trainingList_highlimit],
                            "Training",
                            self.training_exodus_path,
                            "HardExodus")
        
        self.helpers.save_images(test_hardExodusJ[testList_lowerlimit:testList_highlimit],
                            test_names[testList_lowerlimit:testList_highlimit],
                            "Tests",
                            self.test_exodusJ_path,
                            "HardExodus")
        
        self.helpers.save_images(training_hardExodusJ[trainingList_lowerlimit:trainingList_highlimit],
                            training_names[trainingList_lowerlimit:trainingList_highlimit],
                            "Training",
                            self.training_exodusJ_path,
                            "HardExodus")
        
        
        for i in range(0,trainingList_highlimit):
            __, imageholder = cv2.threshold(self.training_groundthruth_dataset[i],5,255,cv2.THRESH_BINARY)
            imageholder = cv2.resize(imageholder,None,fx=0.60,fy=0.60)       
            countours, __ = cv2.findContours(imageholder,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) [-2:]
            cnt, numcountours = self.helpers.evaluate_exodus(training_hardExodus[i],imageholder,self.training_dataset[i])
            
            print("HARD_EXODUS IDRiD{}: Real Exodus: {} Exodus Accepted: {} Percentaje: {:%}".format(i,len(countours),cnt,cnt/len(countours)))
            
            cnt, numcountours = self.helpers.evaluate_exodus(training_hardExodusJ[i],self.training_groundthruth_dataset[i],self.training_dataset[i])

            print("HARD_EXODUSJ IDRiD{}: Real Exodus: {} Exodus Accepted: {} Percentaje: {:%}".format(i,len(countours),cnt,cnt/len(countours)))
            
        
        """Precisions = []
        Recalls = []
        Index = []
        for i in range(0,trainingList_highlimit):
            img_resized = cv2.resize(training_groundthruth_dataset[i],None,fx=0.60,fy=0.60)
            intersect = np.sum(img_resized*training_hardExodus[i])
            total_pixel_truth = np.sum(img_resized)
            Index.append("IDRiD_0{}".format(i+1))
            print("IDRiD_0{}: True Positive: {} | total ground thruth: {} | percentaje: {}".format(i+1,intersect,total_pixel_truth,total_pixel_truth/intersect)) """
    def feature_evaluation(self):

        
        return None


flow = pipeline()

flow.__init__()
flow.preprocessing_featureExtraction()

end_time = time.time()
elapsed_time = end_time - flow.start_time
elapsed_time = elapsed_time/60
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print("Program ended the elapsed time is {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

