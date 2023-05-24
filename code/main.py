import cv2,time,os,argparse
from proj_functions import proj_functions
from preposcessing import preprocessing
from hard_exodus import HardExodus
from feature import feature
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from sklearn.preprocessing import MinMaxScaler

class pipeline():
    def __init__(self):
        self.start_time = time.time()
        self.test_image = np.zeros((2848, 4288, 3), dtype = "uint8")
        self.training_image = np.zeros((2848, 4288, 3),dtype="uint8")
        self.hard_exodus = np.zeros((2848, 4288, 3),dtype="uint8")
        self.parser = argparse.ArgumentParser("Project to detected Hard Exodus")
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
        self.training92_sensivities = []
        self.training97_sensivities = []
        self.full_image_sensivity_92 = []
        self.full_image_sensivity_97 = []
        
    def preprocessing(self):
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
        testList_highlimit,trainingList_highlimit = self.helpers.settingLimits(arguments.highlimit,testList_length,trainingList_length)
        
        #Reading all the images and append it to the empty list
        training_dataset = []
        test_dataset = []
        training_groundthruth_dataset = []
        
        for i in range(0,testList_length):
            
            test_dataset.append(
                cv2.imread(test_path+test_names[i],cv2.COLOR_BGR2RGB)
            )
        
        for i in range(0,trainingList_length):
            
            training_dataset.append(
                cv2.imread(training_path+training_names[i],cv2.COLOR_BGR2RGB)
            )
            
            training_groundthruth_dataset.append(
                cv2.imread(training_groundtruths_path+training_groundtruths_names[i],cv2.IMREAD_UNCHANGED)
            )
        
        test_prepos = preprocessing(
            "Tests",
            test_dataset[testList_lowerlimit:testList_highlimit]
            )
        
        test_greenchannel,test_denoising,test_median = test_prepos.get_Prepocessing()
        
        training_prepos = preprocessing(
            "Training",
            training_dataset[trainingList_lowerlimit:trainingList_highlimit]   
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
        
        print('Saving Variables. Know you can commet this method')
        
        file = open("variable_save/Global_Variables.pickle", "wb")
        pickle.dump(testList_length, file)
        pickle.dump(trainingList_length,file)
        pickle.dump(training_names, file)
        pickle.dump(test_names, file)
        pickle.dump(test_dataset, file)
        pickle.dump(training_dataset, file)
        pickle.dump(training_groundthruth_dataset,file)
        file.close()
        
        file = open("variable_save/prepos_out.pickle","wb")
        pickle.dump(test_denoising, file)
        pickle.dump(training_denoising, file)
        file.close()

        
        
    def hard_exodus_extraction_treshHold(self):

        try:
        # Open the file in binary mode
            with open("variable_save/Global_Variables.pickle", "rb") as file:
                try:
                    # Deserialize and load the variables from the file
                    testList_length = pickle.load(file)
                    trainingList_length = pickle.load(file)
                    training_names = pickle.load(file)
                    test_names = pickle.load(file)
                    print("Pickle file loaded successfully!")
                    file.close()
                except EOFError:
                    print("The pickle file is empty.")
            with open("variable_save/prepos_out.pickle","rb") as file:
                test_denoising = pickle.load(file)
                training_denoising = pickle.load(file)
                file.close()
                
        except FileNotFoundError:
            print("The pickle file does not exist.")
            print("Run the Preposeccing step first")
        
        ##Geting all the the args
        arguments = self.parser.parse_args()
        
        testList_lowerlimit,trainingList_lowerlimit = self.helpers.settingLimits(arguments.lowerlimit,0,0)
        testList_highlimit,trainingList_highlimit = self.helpers.settingLimits(arguments.highlimit,testList_length,trainingList_length)
        
        ###Hard Exodus
        test_Exodus = HardExodus(
            "Test",
            test_denoising[testList_lowerlimit:testList_highlimit])
        
        training_Exodus = HardExodus(
            "Training",
            training_denoising[trainingList_lowerlimit:trainingList_highlimit])
        
        
        test_hard92,test_hard97 = test_Exodus.getHardExodus([92,97])
        training_hard92,training_hard97 = training_Exodus.getHardExodus([92,97])
        
        _Results = [test_hard92,test_hard97,training_hard92,training_hard97]
        _Folder = ['Tests','Tests','Training','Training']
        _SubFolder = ['HardExodus_92','HardExodus_97','HardExodus_92','HardExodus_97']
        _names = [test_names, test_names, training_names, training_names]
        _limitsLow = [testList_lowerlimit,testList_lowerlimit,trainingList_lowerlimit,trainingList_lowerlimit]
        _limitsHigh = [testList_highlimit,testList_highlimit,trainingList_highlimit,trainingList_highlimit]
        
        for i in range(0,4):
            data = _Results[i]
            name = _names[i]
            low = _limitsLow[i]
            high = _limitsHigh[i]
            
            self.helpers.save_images(data[low:high],
                                     name[low:high],
                                     _Folder[i],
                                     os.path.join(self.currentpath,'Results',_SubFolder[i],_Folder[i]),_SubFolder[i])
            
        file = open("variable_save/exodus_out.pickle","wb")
        pickle.dump(test_hard92, file)
        pickle.dump(training_hard92, file)
        pickle.dump(test_hard97, file)
        pickle.dump(training_hard97, file)
        file.close()
        
 
 
    def get_Features(self):
        try:
            # Open the file in binary mode
            with open("variable_save/exodus_out.pickle","rb") as file:
                try:
                    test_hard92 = pickle.load(file)
                    training_hard92 = pickle.load(file)
                    test_hard97 = pickle.load(file)
                    training_hard97 = pickle.load(file)
                    file.close()
                except EOFError:
                    print("The pickle file is empty.")
        
            with open("variable_save/Global_Variables.pickle", "rb") as file:
                try:
                    # Deserialize and load the variables from the file
                    testList_length = pickle.load(file)
                    trainingList_length = pickle.load(file)
                    training_names = pickle.load(file)
                    test_names = pickle.load(file)
                    test_dataset = pickle.load(file)
                    training_dataset = pickle.load(file)
                    training_groundthruth_dataset = pickle.load(file)
                    
                    print("Pickle file loaded successfully!")
                    file.close()
                except EOFError:
                    print("The pickle file is empty.")

        except FileNotFoundError:
            print("The pickle file does not exist.")
            print("Run the Preposeccing step first")
            
        ##Geting all the the args
        arguments = self.parser.parse_args()
        
        testList_lowerlimit,trainingList_lowerlimit = self.helpers.settingLimits(arguments.lowerlimit,0,0)
        testList_highlimit,trainingList_highlimit = self.helpers.settingLimits(arguments.highlimit,testList_length,trainingList_length)
        
        ground92, exodus92, ground97, exodus97 = [], [], [], []
        
        with tqdm(total=trainingList_highlimit,desc="Feature extraction") as pbar:
            for i in range(0,trainingList_highlimit):
                __, imageholder = cv2.threshold(training_groundthruth_dataset[i],5,255,cv2.THRESH_BINARY)
                imageholder = cv2.resize(imageholder,None,fx=0.40,fy=0.40)       
                countours, __ = cv2.findContours(imageholder,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) [-2:]
                
                neg_92, pos_92, training92_sensitivity, image_sensivity_92, exodus_ground92, counted92= self.helpers.evaluate_exodus(training_hard92[i],imageholder,training_dataset[i])
                neg_97, pos_97, training97_sensitivity, image_sensivity_97, exodus_ground97, counted97= self.helpers.evaluate_exodus(training_hard97[i],imageholder,training_dataset[i])
                
                self.training92_sensivities.append(training92_sensitivity)
                self.training97_sensivities.append(training97_sensitivity)
                
                self.full_image_sensivity_92.append(image_sensivity_92)
                self.full_image_sensivity_97.append(image_sensivity_97)
                
                ground92.append(exodus_ground92)
                exodus92.append(counted92)
                
                ground97.append(exodus_ground97)
                exodus97.append(counted97)
                
                df = pd.DataFrame(neg_92)
                
                df = pd.DataFrame(neg_92)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('neg_92.csv',mode='a', index=False, header=False)
                
                df = pd.DataFrame(neg_97)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('neg_97.csv',mode='a', index=False, header=False)
                
                df = pd.DataFrame(pos_92)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('pos_92.csv',mode='a', index=False, header=False)
                
                df = pd.DataFrame(pos_97)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('pos_97.csv',mode='a', index=False, header=False)
                
                pbar.update(1)
                
        for i in range(0, len(self.training97_sensivities)):
            print('IDiD_0{} Sensivities_In_full_image: Percen_92: {:%} Percen_97: {:%}'.format(i+1,self.full_image_sensivity_92[i],self.full_image_sensivity_97[i]))
        print(' ')
        print('Average sensivity: Percen_92 {} Percen_97 {}'.format(sum(self.full_image_sensivity_92)/len(self.full_image_sensivity_92),
                                                                    sum(self.full_image_sensivity_97)/len(self.full_image_sensivity_97)))
        print('***********************************************************************************')
        
        for i in range(0, len(self.training97_sensivities)):    
            print('IDiD_0{} Sensivities_Avg_Selected_Hard_Exodus: Percen_92: {:%} Percen_97: {:%}'.format(i+1,self.training92_sensivities[i],self.training97_sensivities[i]))
        
        print(' ')
        print('Average sensivity Selected_Hard_exodus: Percen_92 {} Percen_97 {}'.format(sum(self.training92_sensivities)/len(self.training92_sensivities),
                                                                    sum(self.training97_sensivities)/len(self.training97_sensivities)))

        print('***********************************************************************************')
        print(len(ground97))
        for i in range(0,len(ground92)):
            print('IDiD_0{} Percentaje of Exodus Recognize: Exodus_92 {:%}'.format(i+1,exodus92[i]/ground92[i]))
            
        print('***********************************************************************************')
        for i in range(0,len(ground97)):
            print('IDiD_0{} Percentaje of Exodus Recognize: Exodus_97 {:%}'.format(i+1,exodus97[i]/ground97[i]))
        
    def normalize_data(self):
        scaler = MinMaxScaler()
        
        neg_92 = pd.read_csv('neg_92.csv')
        pos_92 = pd.read_csv('pos_92.csv')
        
        neg_97 = pd.read_csv('neg_97.csv')
        pos_97 = pd.read_csv('pos_97.csv')
        
        neg_92 = scaler.fit_transform(neg_92)
        neg_97 = scaler.fit_transform(neg_97)
        
        pos_92 = scaler.fit_transform(pos_92)
        pos_97 = scaler.fit_transform(pos_97)
        
        np.savetxt("neg_92.csv",neg_92,delimiter=',')
        np.savetxt("neg_97.csv",neg_97,delimiter=',')
        
        np.savetxt("pos_92.csv",pos_92,delimiter=',')
        np.savetxt("pos_97.csv",pos_97,delimiter=',')
        
flow = pipeline()

flow.__init__()
#flow.preprocessing()
#flow.hard_exodus_extraction_treshHold()
flow.get_Features()
flow.normalize_data()

end_time = time.time()
elapsed_time = end_time - flow.start_time
elapsed_time = elapsed_time/60
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print("Program ended the elapsed time is {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

