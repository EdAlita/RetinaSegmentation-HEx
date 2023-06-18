import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class aurp_curve():
    def __init__(self):
        self.total_groundThruth = 10488
        
    def get_data_UNICASML(self,pos_file):
        data = pd.read_csv('Results/{}'.format(pos_file),delimiter='	')
        data=data.sort_values('#SCORE',ascending=False)
        data = data.round(2)
        return data
    
    def get_data(self,pos_file):
        data = pd.read_csv('Results/{}'.format(pos_file),header=None)
        data = data.iloc[:,1]
        data = data.round(2)
        return data
         
    def calculate_precision_recall_curve(self,positive_scores):
        """
        Calculate the precision-recall curve.

        Parameters:
        positive_scores (array-like): Scores of positive samples.
        
        Returns:
        ndarray: Array of precision values.
        ndarray: Array of recall values.
        """
        thresholds = np.linspace(min(positive_scores),max(positive_scores),100)
        precisions = []
        recalls = []
        
        for threshold in thresholds:
            tp = np.sum(positive_scores >= threshold)
            fp = np.sum(positive_scores < threshold)

            recall  = tp / self.total_groundThruth
            precision = tp / (tp + fp)

            precisions.append(precision)
            recalls.append(recall)
            
        return np.array(precisions), np.array(recalls) 
       
    def get_aurp_curve(self,precisions,recalls,origin,auc):
        plt.figure()
        plt.plot(recalls,precisions, label='AUPR: {:%}'.format(auc))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall {}".format(origin))
        plt.legend(loc='best')
        plt.show()
        plt.figure()
        plt.xlabel("")
        plt.ylabel("Treshold")
        plt.plot(recalls, label = 'Recall')
        plt.plot(precisions, label = 'Presicion')
        plt.legend(loc='best')
        plt.show()
        return None