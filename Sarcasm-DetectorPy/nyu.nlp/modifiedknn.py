
import numpy as np
import pandas as pd
import knnclassifier as knn
from collections import Counter

class ModifiedKNN():

    def count_labels(self,traintarget):
        arr=np.array(traintarget)
        arr.sort()
        count_labels=np.array(10)

        print count_labels
    def predict_each_test(self,trainvectors,traintarget,testvectors):
        for each_test_vector in testvectors:
            dist=np.array(traintarget.len)
            for each_train_vector in trainvectors:
                distance=self.calculate_euclidean(each_test_vector,each_train_vector)
                dist.append(distance)
            dict={'Distance': dist,'Scores' : traintarget}
            df=pd.DataFrame(data=dict)
            df=df.sort_values(by='Distance')


    def calculate_euclidean(self, f1,f2):
        return np.dot(f1,f2)

    def test(self):
        arr=[1,2,3,1,1,1]
        self.count_labels(arr)
def main():
    print ("In main")
    ModifiedKNN().test()
if __name__ == '__main__':
    main()