
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from loguru import logger
import pandas as pd
import pickle


class BinaryClassifierEvaluator:
    def __init__(self, classifiers):
        self.classifiers = classifiers
        self.results = None
        self.best_classifier = None
        self.X_test = None
        self.y_test = None
        logger.info(self.__dict__)
        
    
    def evaluate(self, X_train, y_train):
        # Split the dataset into train and test sets
        #X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Perform grid search for each classifier and store results
        results = []
        for classifier in self.classifiers:
            gs = GridSearchCV(classifier['model'], classifier['params'], scoring='roc_auc', cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1),verbose=10,n_jobs=-1)
            gs.fit(X_train, y_train)
            best_estimator = gs.best_estimator_
            y_pred = best_estimator.predict_proba(self.X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(self.y_test, y_pred)
            auc_score = auc(fpr, tpr)
            results.append({'Classifier': classifier['name'], 'Best Estimator': best_estimator, 'AUC': auc_score})
            
        # Store results and best classifier
        self.results = pd.DataFrame(results)
        self.best_classifier = self.results.loc[self.results['AUC'].idxmax()]
        
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
        best_auc = self.best_classifier['AUC']

        print("Best Classifier:")
        print("  Name:", self.best_classifier['Classifier'])
        print("  Parameters:", best_params)
        print("  AUC:", best_auc)

    
    def plot_roc_curve(self, dataname, X_test, y_test):
        if self.best_classifier is None:
            print("No best classifier found. Please run evaluate() first.")
            return

        plt.figure()
        for index, row in self.results.iterrows():
            name = row['Classifier']
            classifier = row['Best Estimator']
            y_pred_proba = classifier.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label='{} (AUC = {:.2f})'.format(name, roc_auc))

        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig('Train_cross_validation_{}.png'.format(dataname))
            
    