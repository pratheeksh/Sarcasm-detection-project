
import numpy as np
import pandas as pd
import knnclassifier as knn
from collections import Counter

class ModifiedKNN():
    K=2
    def count_labels(self,traintarget):
        arr=np.array(traintarget)
        arr.sort()
        count_labels=Counter(arr)
        count_total=arr.__len__()
        return count_labels,count_total
    # predict

    def predict_test(self,trainvectors,traintarget,testvectors):
        countlabels,count_total=self.count_labels(traintarget)
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
            print topKscores
            for i in range(self.K):
                label_i = int(topKscores[i])
                prediction=prediction+ countlabels[label_i]
            print prediction
            prediction=prediction/self.K
            test_predictions.append(prediction)

        return test_predictions
    def calculate_euclidean(self, f1,f2):
        return np.dot(f1,f2)


def main():
    print "In main"
if __name__ == '__main__':
    main()