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
        self.parser = argparse.ArgumentParser("Project to detected Hard Exodus")
        self.parser.add_argument("-ll","--lowerlimit", default=100,help='Gives the lower limit to cut the data. Default value is the entire array of Data')
        self.parser.add_argument("-lh","--highlimit", default=100,help='Gives the higher limit to cut the data. Default value is the entire array of Data')
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
            "Tests",
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
            "Tests",
            os.path.join(self.currentpath,'Results','Prepos','Tests'),
            "Prepos")
        self.helpers.save_images(
            training_denoising[trainingList_lowerlimit:trainingList_highlimit],
            training_names[trainingList_lowerlimit:trainingList_highlimit],
            "Training",
            os.path.join(self.currentpath,'Results','Prepos','Training'),
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
                    logger.info("Pickle file loaded successfully")
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
        arguments = self.parser.parse_args()
        #setting the limits of the dataset
        testList_lowerlimit,trainingList_lowerlimit = self.helpers.settingLimits(arguments.lowerlimit,0,0)
        testList_highlimit,trainingList_highlimit = self.helpers.settingLimits(arguments.highlimit,testList_length,trainingList_length)
        
        ###Hard Exodus
        test_Exodus = HardExodus(
            "Test",
            test_denoising[testList_lowerlimit:testList_highlimit]
            )
        
        training_Exodus = HardExodus(
            "Training",
            training_denoising[trainingList_lowerlimit:trainingList_highlimit]
            )
        
        
        test_hard92,test_hard97 = test_Exodus.getHardExodus([92,97])
        training_hard92,training_hard97 = training_Exodus.getHardExodus([92,97])
        
        
        #Saving the images of the results
        _Results = [test_hard92,test_hard97,training_hard92,training_hard97]
        _Folder = ['Tests','Tests','Training','Training']
        _SubFolder = ['HardExodus_92','HardExodus_97','HardExodus_92','HardExodus_97']
        _names = [test_names,test_names,training_names,training_names]
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
            
        logger.info('Saving Variables. Know you can commet this method')
            
        file = open("variable_save/exodus_out.pickle","wb")
        pickle.dump(test_hard92, file)
        pickle.dump(training_hard92, file)
        pickle.dump(test_hard97, file)
        pickle.dump(training_hard97, file)
        file.close()
        
 
 
    def get_Features(self):
        #getting the store values from the flow
        try:
            with open("variable_save/exodus_out.pickle","rb") as file:
                try:
                    test_hard92 = pickle.load(file)
                    training_hard92 = pickle.load(file)
                    test_hard97 = pickle.load(file)
                    training_hard97 = pickle.load(file)
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
                    
                    logger.info("Pickle file loaded successfully!")
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
        ground92, exodus92, ground, exodus97, training92_sensivities, training97_sensivities, test92_sensivities, test97_sensivities, y_output  = [], [], [], [], [], [] , [], [], []
        
        #Extracting the features trainnig data set
        with tqdm(total=trainingList_highlimit,desc="Feature extraction training data set") as pbar:
            for i in range(0,trainingList_highlimit):
                __, imageholder2 = cv2.threshold(training_groundthruth_dataset[i],0,255,cv2.THRESH_BINARY)
                imageholder2 = cv2.resize(imageholder2,None,fx=0.40,fy=0.40)       
                
                neg_92, pos_92, training92_sensitivity, exodus_ground92, counted92, y_92n, y_92p= self.helpers.evaluate_exodus(training_hard92[i],
                                                                                                                 imageholder2,
                                                                                                                 training_dataset[i],
                                                                                                                 i)
                neg_97, pos_97, training97_sensitivity, exodus_ground97, counted97, y_97n, y_97p = self.helpers.evaluate_exodus(training_hard97[i],
                                                                                                                 imageholder2,
                                                                                                                 training_dataset[i],
                                                                                                                 i)
                
                training92_sensivities.append(training92_sensitivity)
                training97_sensivities.append(training97_sensitivity)
                
                #ground.append(exodus_ground92)
                exodus92.append(counted92)
                
                ground.append(exodus_ground97)
                exodus97.append(counted97)
                
                
                df = pd.DataFrame(neg_92)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('neg_92.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(y_92n)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Yneg_92.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(neg_97)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('neg_97.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(y_97n)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Yneg_97.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(pos_92)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('pos_92.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(y_92p)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Ypos_92.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(pos_97)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('pos_97.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(y_97p)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Ypos_97.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                pbar.update(1)
        #Extracting the data set
        
        with tqdm(total=testList_length,desc="Feature extraction test data set") as pbar:
            for i in range(0,testList_length):
                __, imageholder = cv2.threshold(test_groundthruth_dataset[i],0,255,cv2.THRESH_BINARY)
                imageholder = cv2.resize(imageholder,None,fx=0.40,fy=0.40)       
                
                neg_92, pos_92, test92_sensitivity, exodus_ground92, counted92, y_92n, y_92p= self.helpers.evaluate_exodus(test_hard92[i],
                                                                                                             imageholder,
                                                                                                             test_dataset[i],
                                                                                                             55+i)
                neg_97, pos_97, test97_sensitivity, exodus_ground97, counted97, y_97n, y_97p= self.helpers.evaluate_exodus(test_hard97[i],
                                                                                                             imageholder,
                                                                                                             test_dataset[i],
                                                                                                             55+i)
                
                test92_sensivities.append(test92_sensitivity)
                test97_sensivities.append(test97_sensitivity)
                
                #ground.append(exodus_ground92)
                exodus92.append(counted92)
                
                ground.append(exodus_ground97)
                exodus97.append(counted97)
                
                df = pd.DataFrame(neg_92)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('neg_92.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(y_92n)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Yneg_92.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(neg_97)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('neg_97.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(y_97n)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Yneg_97.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(pos_92)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('pos_92.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(y_92p)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Ypos_92.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(pos_97)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('pos_97.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                df = pd.DataFrame(y_97p)
                df = df.applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)
                df.to_csv('Ypos_97.csv',mode='a', index=False, header=False,float_format='%.15f')
                
                pbar.update(1)
        
                
        logger.info('Saving variables...')
        file = open("variable_save/get_exodus_out.pickle","wb")
        pickle.dump(training92_sensivities, file)
        pickle.dump(training97_sensivities, file)
        pickle.dump(test92_sensivities,file)
        pickle.dump(test97_sensivities,file)
        pickle.dump(exodus92,file)
        pickle.dump(exodus97,file)
        pickle.dump(ground92,file)
        pickle.dump(ground,file)
        print(sum(ground))
        file.close()            
            
                

    
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
        
        neg_92 = pd.read_csv('neg_92.csv')
        pos_92 = pd.read_csv('pos_92.csv')
        
        neg_97 = pd.read_csv('neg_97.csv')
        pos_97 = pd.read_csv('pos_97.csv')
        
        neg_92 = scaler.fit_transform(neg_92)
        neg_97 = scaler.fit_transform(neg_97)
        
        pos_92 = scaler.fit_transform(pos_92)
        pos_97 = scaler.fit_transform(pos_97)
        
        np.savetxt("neg_92.csv",neg_92,fmt="%f",delimiter=',')
        np.savetxt("neg_97.csv",neg_97,fmt="%f",delimiter=',')
        
        np.savetxt("pos_92.csv",pos_92,fmt="%f",delimiter=',')
        np.savetxt("pos_97.csv",pos_97,fmt="%f",delimiter=',')
        
    def ML(self,
           dataset):
        logger.info('Creating the Machine Learning for {}'.format(dataset))
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        negative_data = pd.read_csv('neg_{}.csv'.format(dataset),header=None)
        positive_data = pd.read_csv('pos_{}.csv'.format(dataset),header=None)
        
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
            {'name': 'k-Nearest Neighbors','model': KNeighborsClassifier(), 'params': param_grid_knn},
            {'name': 'Decision Tree', 'model': DecisionTreeClassifier(), 'params': param_grid_dt},
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



