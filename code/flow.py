import cv2,time,os,argparse
from proj_functions import proj_functions
from preposcessing import preprocessing
from hard_exodus import HardExodus
from feature import feature
from loguru import logger
from machine_learning import BinaryClassifierEvaluator
import numpy as np
import pandas as pd
from tqdm import tqdm
from alive_progress import alive_bar
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings 
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


class pipeline():
    def __init__(self):
        """Init for the pipeline of the code
        """
        #self.parser = argparse.ArgumentParser("Project to detected Hard Exodus")
        #self.parser.add_argument("-ll","--lowerlimit", default=100,help='Gives the lower limit to cut the data. Default value is the entire array of Data')
        #self.parser.add_argument("-lh","--highlimit", default=100,help='Gives the higher limit to cut the data. Default value is the entire array of Data')
        self.currentpath = os.getcwd()
        self.helpers = proj_functions()
        
        logger.info(f"Class Initialized: {self.__dict__}")
        
    def preprocessing(self):
        """Preprocessing of the images
        """
        #Empty list to save the data
        training_dataset, test_dataset, training_groundthruth_dataset, test_groundthruth_dataset = [], [], [], []
        #Geting all the the args
        #arguments = self.parser.parse_args()
        arguments = 100
        self.helpers.file_structure()
        
        #Open file to obtain local path to the data field
        test_path,training_path,training_groundtruths_path,test_groundtruths_path = self.helpers.get_localDirectories()
        
        #Grabing all the names of the Data.
        test_names= os.listdir(test_path)
        training_names= os.listdir(training_path)
        training_groundtruths_names=os.listdir(training_groundtruths_path)
        test_groundtruths_names = os.listdir(test_groundtruths_path)
        
        #sorting the images
        test_names.sort()
        training_names.sort()
        training_groundtruths_names.sort()
        test_groundtruths_names.sort()
        
        #Getting length of the data
        testList_length = 25
        trainingList_length = 54
        
        #Setting the limits for the data
        testList_lowerlimit,trainingList_lowerlimit = self.helpers.settingLimits(arguments,0,0)
        testList_highlimit,trainingList_highlimit = self.helpers.settingLimits(arguments,testList_length,trainingList_length)
        
        #Reading all the images and append it to the empty list
        for i in range(0,testList_length):
            test_dataset.append(
                cv2.imread(test_path+test_names[i],cv2.COLOR_BGR2RGB)
            )
            test_groundthruth_dataset.append(
                cv2.imread(test_groundtruths_path+test_groundtruths_names[i],cv2.IMREAD_UNCHANGED)
            )
        
        for i in range(0,trainingList_length):
            training_dataset.append(
                cv2.imread(training_path+training_names[i],cv2.COLOR_BGR2RGB)
            )
            training_groundthruth_dataset.append(
                cv2.imread(training_groundtruths_path+training_groundtruths_names[i],cv2.IMREAD_UNCHANGED)
            )
        
        #Get all the preprocesing of the data
        test_prepos = preprocessing(
            "Test",
            test_dataset[testList_lowerlimit:testList_highlimit]
            )
        training_prepos = preprocessing(
            "Training",
            training_dataset[trainingList_lowerlimit:trainingList_highlimit]   
        )
        test_denoising = test_prepos.get_Prepocessing()
        training_denoising = training_prepos.get_Prepocessing()
        
        #Saving the Images
        self.helpers.save_images(
            test_denoising[testList_lowerlimit:testList_highlimit],
            test_names[testList_lowerlimit:testList_highlimit],
            "Test",
            os.path.join(self.currentpath,'Results','Prepos','Test'),
            "Prepos")
        self.helpers.save_images(
            training_denoising[trainingList_lowerlimit:trainingList_highlimit],
            training_names[trainingList_lowerlimit:trainingList_highlimit],
            "Train",
            os.path.join(self.currentpath,'Results','Prepos','Train'),
            "Prepos")
        
        #Saving a pickle of the variables
        logger.info('Saving Variables. Know you can commet this method')
        
        file = open("variable_save/Global_Variables.pickle", "wb")
        pickle.dump(testList_length, file)
        pickle.dump(trainingList_length,file)
        pickle.dump(training_names, file)
        pickle.dump(test_names, file)
        pickle.dump(test_dataset, file)
        pickle.dump(training_dataset, file)
        pickle.dump(training_groundthruth_dataset,file)
        pickle.dump(test_groundthruth_dataset, file)
        file.close()
        
        file = open("variable_save/prepos_out.pickle","wb")
        pickle.dump(test_denoising, file)
        pickle.dump(training_denoising, file)
        file.close()
        
        logger.success('Exit without errors')

        
        
    def hard_exodus_extraction_treshHold(self):
        """Extracting the hard exodus with threshHolding and binarization
        """
        #Getting the variables that are save
        try:
            with open("variable_save/Global_Variables.pickle", "rb") as file:
                try:
                    testList_length = pickle.load(file)
                    trainingList_length = pickle.load(file)
                    training_names = pickle.load(file)
                    test_names = pickle.load(file)
                    logger.success("Pickle file loaded successfully")
                    file.close()
                except EOFError:
                    logger.error("The pickle file is empty.")
            with open("variable_save/prepos_out.pickle","rb") as file:
                test_denoising = pickle.load(file)
                training_denoising = pickle.load(file)
                file.close()
                
        except FileNotFoundError:
            logger.error("The pickle file does not exist.")
            logger.error("Run the Preposeccing step first")
        
        ##Geting all the the args
        arguments = 100
        #setting the limits of the dataset
        testList_lowerlimit,trainingList_lowerlimit = self.helpers.settingLimits(arguments,0,0)
        testList_highlimit,trainingList_highlimit = self.helpers.settingLimits(arguments,testList_length,trainingList_length)
        
        ###Hard Exodus
        test_Exodus = HardExodus(
            "Test",
            test_denoising[testList_lowerlimit:testList_highlimit]
            )
        
        training_Exodus = HardExodus(
            "Training",
            training_denoising[trainingList_lowerlimit:trainingList_highlimit]
            )
        
        
        test_hard85,test_hard90, test_hard95  = test_Exodus.getHardExodus([95,90,95])
        training_hard85,training_hard90, training_hard95 = training_Exodus.getHardExodus([95,90,95])
        
        
        #Saving the images of the results
        _Results = [test_hard85,test_hard90, test_hard95,training_hard85,training_hard90, training_hard95]
        _Folder = ['Test','Test','Test','Train','Train','Train']
        _SubFolder = ['HardExodus_85','HardExodus_90','HardExodus_95','HardExodus_85','HardExodus_90','HardExodus_95']
        _names = [test_names,test_names,test_names,training_names,training_names,training_names]
        _limitsLow = [testList_lowerlimit,testList_lowerlimit,testList_lowerlimit,trainingList_lowerlimit,trainingList_lowerlimit,trainingList_lowerlimit]
        _limitsHigh = [testList_highlimit,testList_highlimit,testList_highlimit,trainingList_highlimit,trainingList_highlimit,trainingList_highlimit]
        
        for i in range(0,6):
            data = _Results[i]
            name = _names[i]
            low = _limitsLow[i]
            high = _limitsHigh[i]
            
            self.helpers.save_images(data[low:high],
                                     name[low:high],
                                     _Folder[i],
                                     os.path.join(self.currentpath,'Results',_SubFolder[i],_Folder[i]),_SubFolder[i])
            
        logger.info('Saving Variables. Know you can commet this method')
            
        file = open("variable_save/exodus_out.pickle","wb")
        pickle.dump(test_hard85, file)
        pickle.dump(training_hard85, file)
        pickle.dump(test_hard90, file)
        pickle.dump(training_hard90, file)
        pickle.dump(test_hard95, file)
        pickle.dump(training_hard95, file)
        file.close()
        
        logger.success('Exit without errors')
        
 
 
    def get_Features(self):
        #getting the store values from the flow
        try:
            with open("variable_save/exodus_out.pickle","rb") as file:
                try:
                    test_hard85 = pickle.load(file)
                    training_hard85 = pickle.load(file)
                    test_hard90 = pickle.load(file)
                    training_hard90 = pickle.load(file)
                    test_hard95 = pickle.load(file)
                    training_hard95 = pickle.load(file)
                    file.close()
                except EOFError:
                    logger.error("The pickle file is empty.")
        
            with open("variable_save/Global_Variables.pickle", "rb") as file:
                try:
                    testList_length = pickle.load(file)
                    trainingList_length = pickle.load(file)
                    __ = pickle.load(file)
                    __ = pickle.load(file)
                    test_dataset = pickle.load(file)
                    training_dataset = pickle.load(file)
                    training_groundthruth_dataset = pickle.load(file)
                    test_groundthruth_dataset = pickle.load(file)
                    
                    logger.success("Pickle file loaded successfully!")
                    file.close()
                except EOFError:
                    logger.error("The pickle file is empty.")

        except FileNotFoundError:
            logger.error("The pickle file does not exist.")
            logger.error("Run the Preposeccing step first")
            
        ##Geting all the the args
        arguments = 100
        #Setting the limits for the data
        testList_highlimit,trainingList_highlimit = self.helpers.settingLimits(arguments,testList_length,trainingList_length)
        ground, exodus85,exodus90,exodus95, training85_sensivities, training90_sensivities, training95_sensivities,test85_sensivities, test90_sensivities,test95_sensivities, y_output  = [], [], [], [], [], [] , [], [], [], [], []
        
        #Extracting the features trainnig data set
        with tqdm(total=trainingList_highlimit,desc="Feature extraction training data set") as pbar:
            for i in range(0,trainingList_highlimit):
                __, imageholder2 = cv2.threshold(training_groundthruth_dataset[i],0,255,cv2.THRESH_BINARY)
                imageholder2 = cv2.resize(imageholder2,None,fx=0.40,fy=0.40)       
                
                neg_85, pos_85, training85_sensitivity, exodus_ground85, counted85, y_85n, y_85p= self.helpers.evaluate_exodus(training_hard85[i],
                                                                                                                 imageholder2,
                                                                                                                 training_dataset[i],
                                                                                                                 i+1,
                                                                                                                 85)
                neg_90, pos_90, training90_sensitivity, exodus_ground90, counted90, y_90n, y_90p = self.helpers.evaluate_exodus(training_hard90[i],
                                                                                                                 imageholder2,
                                                                                                                 training_dataset[i],
                                                                                                                 i+1,
                                                                                                                 90)
                
                neg_95, pos_95, training95_sensitivity, exodus_ground95, counted95, y_95n, y_95p = self.helpers.evaluate_exodus(training_hard95[i],
                                                                                                                 imageholder2,
                                                                                                                 training_dataset[i],
                                                                                                                 i+1,
                                                                                                                 95)
                
                training85_sensivities.append(training85_sensitivity)
                training90_sensivities.append(training90_sensitivity)
                training95_sensivities.append(training95_sensitivity)
                exodus85.append(counted85)
                exodus90.append(counted90)
                exodus95.append(counted95)
                
                ground.append(exodus_ground90)
                
                df = pd.DataFrame(neg_85)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Results/neg.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(y_85n)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Results/Yneg.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(neg_90)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Results/neg.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(y_90n)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Results/Yneg.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(neg_95)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Results/neg.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(y_95n)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Results/Yneg.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(pos_85)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Results/pos.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(y_85p)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Results/Ypos.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(pos_90)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Results/pos.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(y_90p)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Results/Ypos.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(pos_95)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Results/pos.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(y_95p)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Results/Ypos.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                pbar.update(1)
        #Extracting the data set
        
        with tqdm(total=testList_length,desc="Feature extraction test data set") as pbar:
            for i in range(0,testList_length):
                __, imageholder = cv2.threshold(test_groundthruth_dataset[i],0,255,cv2.THRESH_BINARY)
                imageholder = cv2.resize(imageholder,None,fx=0.40,fy=0.40)       
                
                neg_85, pos_85, test85_sensitivity, exodus_ground85, counted85, y_85n, y_85p= self.helpers.evaluate_exodus(test_hard85[i],
                                                                                                             imageholder,
                                                                                                             test_dataset[i],
                                                                                                             55+i,
                                                                                                             85)
                neg_90, pos_90, test90_sensitivity, exodus_ground90, counted90, y_90n, y_90p= self.helpers.evaluate_exodus(test_hard90[i],
                                                                                                             imageholder,
                                                                                                             test_dataset[i],
                                                                                                             55+i,
                                                                                                             90)
                
                neg_95, pos_95, test95_sensitivity, exodus_ground95, counted95, y_95n, y_95p= self.helpers.evaluate_exodus(test_hard95[i],
                                                                                                             imageholder,
                                                                                                             test_dataset[i],
                                                                                                             55+i,
                                                                                                             95)
                
                test85_sensivities.append(test85_sensitivity)
                test90_sensivities.append(test90_sensitivity)
                test95_sensivities.append(test95_sensitivity)
                
                #ground.append(exodus_ground92)
                exodus85.append(counted85)
                exodus90.append(counted90)
                exodus95.append(counted95)
                
                ground.append(exodus_ground90)
                
                df = pd.DataFrame(neg_85)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Results/neg.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(y_85n)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Results/Yneg.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(neg_90)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Results/neg.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(y_90n)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Results/Yneg.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(neg_95)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Results/neg.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(y_95n)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Results/Yneg.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(pos_85)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Results/pos.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(y_85p)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Results/Ypos.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(pos_90)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Results/pos.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(y_90p)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Results/Ypos.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(pos_95)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Results/pos.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(y_95p)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Results/Ypos.csv',mode='a', index=False, header=False,float_format='%.15f')
        
                pbar.update(1)
        
                
        logger.info('Saving variables...')
        file = open("variable_save/get_exodus_out.pickle","wb")
        pickle.dump(training85_sensivities, file)
        pickle.dump(training90_sensivities, file)
        pickle.dump(training95_sensivities, file)
        pickle.dump(test85_sensivities,file)
        pickle.dump(test90_sensivities,file)
        pickle.dump(test95_sensivities,file)
        pickle.dump(exodus85,file)
        pickle.dump(exodus90,file)
        pickle.dump(exodus95,file)
        pickle.dump(ground,file)
        file.close()
        gTh = sum(ground)
        exodus = sum(exodus85) + sum(exodus90) + sum(exodus95)
        logger.info('Number of Exodus in Groundthruth: {}'.format(gTh))
        logger.info('Exodus counted in segmentation process: {}'.format(exodus)) 
        logger.info('Percentaje recognize {:%}'.format(exodus/gTh))
        logger.info('Exodus obtain wth Tophat Th 95: {}'.format(sum(exodus85))) 
        logger.info('Exodus obtain wth Morph Smoothing Th 90: {}'.format(sum(exodus90))) 
        logger.info('Exodus obtain wth Morph Smoothing Th 95: {}'.format(sum(exodus95)))
        
        logger.success('Terminated without any Error') 

            
                

    
    def print_data(self):
        
        for i in range(0, len(self.training97_sensivities)):    
            print('IDiD_0{} Sensivities_Avg_Selected_Hard_Exodus: Percen_92: {:%} Percen_97: {:%}'.format(
                i+1,
                training92_sensivities[i],
                training97_sensivities[i]))
        
        print('Average sensivity Selected_Hard_exodus: Percen_92 {:%} Percen_97 {:%}'.format(
            sum(training92_sensivities)/len(training92_sensivities),
            sum(training97_sensivities)/len(training97_sensivities)))

        for i in range(0,len(ground92)):
            print('0{} Percentaje of Exodus Recognize: Exodus_92 {:%}'.format(
                i+1,
                exodus92[i]/ground92[i]))
            
        for i in range(0,len(ground97)):
            print('0{} Percentaje of Exodus Recognize: Exodus_97 {:%}'.format(
                i+1,
                exodus97[i]/ground97[i]))
        
        
    def normalize_data(self):
        scaler = MinMaxScaler()
        
        
        try:
            neg = pd.read_csv('Results/neg.csv',header=None)
            pos = pd.read_csv('Results/pos.csv',header=None)
            logger.success("Reading the CVS Files Succesfully")
        except pd.errors.ParserError as e:
            logger.error("Error ocurred while reading CSV Files:")
            return
        
        merge_data = pd.concat([neg, pos],axis=0)
        
        normalize_data = scaler.fit_transform(merge_data)
        
        
        np.savetxt("Results/neg.csv",normalize_data[:len(neg)],fmt="%f",delimiter=',')
        
        np.savetxt("Results/pos.csv",normalize_data[len(neg):],fmt="%f",delimiter=',')
        
        logger.success("Terminated without any Errors")
               
    def ML(self,
           dataset):
        logger.info('Creating the Machine Learning for {}'.format(dataset))
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        negative_data = pd.read_csv('Results/neg.csv',header=None)
        positive_data = pd.read_csv('Results/pos.csv',header=None)
        
        negative_data = negative_data.reset_index(drop=True)
        positive_data = positive_data.reset_index(drop=True)
        print("Columns of X_train:")
        print(negative_data.columns)

        print("\nColumns of X_test:")
        print(positive_data.columns)
        
        # Combine negative and positive data into X_train and y_train
        X_train = pd.concat([negative_data, positive_data],axis=0)
        y_train = pd.concat([pd.Series(np.zeros(len(negative_data))), pd.Series(np.ones(len(positive_data)))], axis=0)
        
        
        # Define the parameter grid for each classifier
        param_grid_lr = {'C': [0.01, 0.1, 1, 10, 100], 'max_iter': [1000000]}
        param_grid_rf = {'n_estimators': [50, 100, 200], 'random_state': [42]}
        param_grid_gb = {'n_estimators': [50, 100, 200], 'learning_rate': [0.1, 0.01], 'max_depth': [3, 5, 10]}
        param_grid_svm = {'gamma': [0.5],'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        param_grid_nb = {'var_smoothing': [1e-9, 1e-8, 1e-7],}
        param_grid_knn = {'n_neighbors': [3,4,5,7],'weights': ['uniform', 'distance'],'algorithm': ['ball_tree', 'kd_tree']}
        param_grid_dt = {'criterion': ['gini', 'entropy'], 'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}



        # Create a list of classifiers to evaluate
        classifiers = [
            {'name': 'Logistic Regression', 'model': LogisticRegression(), 'params': param_grid_lr},
            {'name': 'Random Forest', 'model': RandomForestClassifier(), 'params': param_grid_rf},
            {'name': 'Gradient Boosting', 'model': GradientBoostingClassifier(), 'params': param_grid_gb},
            {'name': 'SVM', 'model': SVC(probability=True), 'params': param_grid_svm},
            {'name': 'Naive Bayes','model': GaussianNB(), 'params': param_grid_nb},
            {'name': 'Decision Tree', 'model': DecisionTreeClassifier(), 'params': param_grid_dt}
            ]

        # Create an instance of the evaluator
        evaluator = BinaryClassifierEvaluator(classifiers,X_train,y_train)

        # Run evaluation on the dataset
        evaluator.evaluate(dataset)

        # Print the results
        #print(evaluator.results)
        
        # Print information about the best classifier
        evaluator.print_best_classifier_info()

        # Plot ROC curve for the best classifier
        #evaluator.plot_roc_curve('97',X_test,Y_test.values.ravel())
        
        evaluator.plot_roc_curve_training(dataset)
        
        #evaluator.plot_roc_curve_test_data('97',X_test)



