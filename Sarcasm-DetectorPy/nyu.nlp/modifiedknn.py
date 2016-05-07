
import numpy as np
import pandas as pd
import knnclassifier as knn
from collections import Counter

class ModifiedKNN():
    K=100
    def count_labels(self,traintarget):
        arr=np.array(traintarget)
        arr.sort()
        count_labels=Counter(arr)
        return count_labels
    # predict
    def predict_test(self,trainvectors,traintarget,testvectors):
        countlabels=self.count_labels(traintarget)
        test_predictions=[]
        for each_test_vector in testvectors:
            dist=[]
            for each_train_vector in trainvectors:
                distance=self.calculate_euclidean(each_test_vector,each_train_vector)
                dist.append(distance)
            dict={'Distance': dist,'Scores' : traintarget}
            df=pd.DataFrame(data=dict)
            df=df.sort_values(by='Distance',ascending=True)[:self.K]
            prediction=0
            topKscores = df['Scores'].tolist()

            for i in range(self.K):
                label_i = int(topKscores[i])
                prediction=prediction+ countlabels[label_i]/100
            prediction=prediction/self.K
            test_predictions.append(prediction)

        return test_predictions
    def calculate_euclidean(self, f1,f2):
        return np.dot(f1,f2)


def main():
    print "In main"
if __name__ == '__main__':
    main()