import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import KFold 
from bayesClass2 import *

#-------------------------------------------------
fold = 4
#-------------------------------------------------

def predictionAndReal(classes,data_set):
    test_array = data_set.to_numpy()
    predict = []
    real = []
    for test_item in test_array:
        maxLikelihood = None
        classNameMax = None
        for classItem in classes:
            likelihood = classItem.calculatLikelihood(test_item[0:4:])
            if maxLikelihood is None or maxLikelihood < likelihood:
                maxLikelihood = likelihood
                classNameMax = classItem.className
        predict.append(classNameMax)
        real.append(test_item[4])
    return predict, real
data = pd.read_csv("iris.data", header=None)
columns = ['x1', 'x2', 'x3', 'x4', 'y']
data.columns = columns
y = data['y']
accuracy_sum_test, accuracy_sum_train = 0, 0

kf = KFold(n_splits=fold, random_state=None, shuffle=True) 
kf.get_n_splits(data) 

counter = 0
for train_index, test_index in kf.split(data):
    counter += 1
    data_train, data_test = data.loc[train_index], data.loc[test_index]
    x1 = data_train.groupby('y')['x1'].apply(list)
    x2 = data_train.groupby('y')['x2'].apply(list)
    x3 = data_train.groupby('y')['x3'].apply(list)
    x4 = data_train.groupby('y')['x4'].apply(list)
    classes = []
    for className, item1, item2, item3, item4 in zip(x1.index, x1, x2, x3, x4):
        bayesClass = bayesClass1(className, item1, item2, item3, item4, len(data_train))
        classes.append(bayesClass)
        
    prediction, reality = predictionAndReal(classes, data_test)
    acc = accuracy_score(reality, prediction)
    accuracy_sum_test += acc
    print('accuracy test = ' + str(acc) + ' counter = ' + str(counter) )
    print('confusion matrix (test) counter = ' + str(counter))
    print(confusion_matrix(reality, prediction))
    
    prediction, reality = predictionAndReal(classes, data_train)
    acc = accuracy_score(reality, prediction)
    accuracy_sum_train += acc
    print('accuracy train = ' + str(acc) + ' counter = ' + str(counter) )
    print('confusion matrix (train) counter = ' + str(counter))
    print(confusion_matrix(reality, prediction))
    
print('--------------------------------------------------')
print('average accuracy test = ' + str(accuracy_sum_test / counter))
print('average accuracy train = ' + str(accuracy_sum_train / counter))

    
    
    
    