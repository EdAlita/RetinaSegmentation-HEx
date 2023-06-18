
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
        self.X_train = X
        self.y_train = Y
        logger.info(self.__dict__)
        
    
    def evaluate(self,dataset,test_size=0.2, random_state=123):
        # Split the dataset into train and test sets
        #X_train, self.X_test, y_train, self.y_test = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state)
        
        # Perform grid search for each classifier and store results
        results = []
        for classifier in self.classifiers:
            gs = GridSearchCV(classifier['model'], classifier['params'], scoring='accuracy', cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=42),verbose=3,n_jobs=-1)
            gs.fit(self.X_train, self.y_train)
            best_estimator = gs.best_estimator_
            y_pred = best_estimator.predict_proba(self.X_train)[:, 1]
            precision, recall, _ = precision_recall_curve(self.y_train, y_pred)
            auc_score = auc(recall, precision)
            results.append({'Classifier': classifier['name'], 'Best Estimator': best_estimator, 'AUC': auc_score})
            
        # Store results and best classifier
        self.results = pd.DataFrame(results)
        self.best_classifier = self.results.loc[self.results['AUC'].idxmax()]
        
        # Split the probabilities into negatives and positives
        negative_probabilities = self.best_classifier['Best Estimator'].predict_proba(self.X_train[self.y_train == 0])
        positive_probabilities = self.best_classifier['Best Estimator'].predict_proba(self.X_train[self.y_train == 1])

        positive_probabilities_df = pd.DataFrame(positive_probabilities)
        negative_probabilities_df = pd.DataFrame(negative_probabilities)

        # Save the DataFrame to a CSV file
        negative_probabilities_df.to_csv('negative_probabilities{}.csv'.format(dataset), index=False)
        positive_probabilities_df.to_csv('positive_probabilities{}.csv'.format(dataset), index=False)

        
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
            y_pred_proba = classifier.predict_proba(self.X_train)[:, 1]
            precision, recall, _ = precision_recall_curve(self.y_train, y_pred_proba)
            roc_auc = auc(recall, precision)
            plt.plot(recall, precision, label='{} (AUPR = {:.2f})'.format(name, roc_auc))

        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precissions')
        plt.title('data_set_{}'.format(dataname))
        plt.legend(loc="lower right")
        plt.savefig('Results/data_set_{}.png'.format(dataname))
            
    