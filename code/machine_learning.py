
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt
from loguru import logger
import pandas as pd
import numpy as np
import pickle


class BinaryClassifierEvaluator:
    def __init__(self, classifiers, X, Y):
        self.classifiers = classifiers
        self.results = None
        self.best_classifier = None
        self.X_test = X
        self.y_test = Y
        logger.info(self.__dict__)
        
    
    def evaluate(self, X_train, y_train, test_size=0.2, random_state=123):
        # Split the dataset into train and test sets
        #X_train, self.X_test, y_train, self.y_test = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state)
        
        # Perform grid search for each classifier and store results
        results = []
        for classifier in self.classifiers:
            gs = GridSearchCV(classifier['model'], classifier['params'], scoring='average_precision', cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=42),verbose=10,n_jobs=-1)
            gs.fit(X_train, y_train)
            best_estimator = gs.best_estimator_
            y_pred = best_estimator.predict_proba(self.X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(self.y_test, y_pred)
            auc_score = auc(recall, precision)
            results.append({'Classifier': classifier['name'], 'Best Estimator': best_estimator, 'AUPR': auc_score})
            
        # Store results and best classifier
        self.results = pd.DataFrame(results)
        self.best_classifier = self.results.loc[self.results['AUPR'].idxmax()]
        
        y = self.best_classifier['Best Estimator'].predict(self.X_test)
        proba = self.best_classifier['Best Estimator'].predict_proba(self.X_test)
        
        print(proba)
        
        predictions = {
            'Predictions': y,
            'Probabilities': proba
        }
        
        result = pd.DataFrame(predictions)
        
        result.to_csv('ML_results.csv')
        
        logger.info('Saving best Classifiier...')
        file = open("variable_save/best_classifier97.pickle","wb")
        pickle.dump(self.best_classifier,file)
        file.close()
        
    def print_best_classifier_info(self):
        if self.best_classifier is None:
            print("No best classifier found. Please run evaluate() first.")
            return

        best_clf = self.best_classifier['Best Estimator']
        best_params = best_clf.get_params()
        best_auc = self.best_classifier['AUPR']

        print("Best Classifier:")
        print("  Name:", self.best_classifier['Classifier'])
        print("  Parameters:", best_params)
        print("  AUPR:", best_auc)

    
    def plot_roc_curve_training(self, dataname):
        if self.best_classifier is None:
            print("No best classifier found. Please run evaluate() first.")
            return

        plt.figure()
        for index, row in self.results.iterrows():
            name = row['Classifier']
            classifier = row['Best Estimator']
            y_pred_proba = classifier.predict_proba(self.X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
            roc_auc = auc(recall, precision)
            plt.plot(recall, precision, label='{} (AUPR = {:.2f})'.format(name, roc_auc))

        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('data_set_{}'.format(dataname))
        plt.legend(loc="lower right")
        plt.savefig('data_set_{}.png'.format(dataname))
            
    