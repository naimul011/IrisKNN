
# coding: utf-8

#Naimul Haque 14.02.04.080

# In[129]:


import numpy as np
import pandas as pd
import operator

# Importing the dataset
dataset = pd.read_csv('Iris.csv')
dataset.shape

dataset.head(5)


# In[130]:


feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']
X = dataset[feature_columns].values
y = dataset['Species'].values


# In[131]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# In[132]:


#My implementaion of the Knn clasifier

def calcDistance(inX, dataSet, labels, k):
    
    dataSetSize = dataSet.shape[0]  

    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet

    sqDiffMat = diffMat ** 2

    sqDistances = sqDiffMat.sum(axis=1)

    distances = sqDistances ** 0.5


    sortedDistIndices = distances.argsort()
    return sortedDistIndices

def majority_element(num_list):
        idx, ctr = 0, 1
        
        for i in range(1, len(num_list)):
            if num_list[idx] == num_list[i]:
                ctr += 1
            else:
                ctr -= 1
                if ctr == 0:
                    idx = i
                    ctr = 1
        
        return num_list[idx]

def findMajorityClass(inX, dataSet, labels, k, sortedDistIndices):
    classNeigbors = []
    
    for i in range(k):
        
        voteIlabel = labels[sortedDistIndices[i]]
        
        classNeigbors.append(voteIlabel)

    return majority_element(classNeigbors)


def classify(inX, dataSet, labels, k):
    
    sortedDistIndices = calcDistance(inX, dataSet, labels, k)
    
    mojorityClass = findMajorityClass(inX, dataSet, labels, k, sortedDistIndices)
    
    return mojorityClass


# In[133]:


#trainng and testing with my knn classifier
def predictWithKnn(testset,dataset,labels,k):
    
    prediction = []
    
    for inx in testset:
        prediction.append(classify(inx,dataset,labels,k))
    return prediction


# In[134]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[135]:


#implementation of sklearn knn clasifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Instantiate learning model (k = 3)
knnClassifier = KNeighborsClassifier(n_neighbors=3)


# Fitting the model
knnClassifier.fit(X_train, y_train)

#prediction with my sklearn knn
pred = knnClassifier.predict(X_test)

#prediction with my knn
mypred = predictWithKnn(X_test,X_train,y_train,3)


# In[136]:
print()
print()

from sklearn.metrics import confusion_matrix, accuracy_score
sklearn_knn = confusion_matrix(y_test, pred)
sklearn_knn

my_knn = confusion_matrix(y_test, mypred)
my_knn

print('Accuracy of sklearn KNN model is equal ' + str(round(accuracy_score(y_test, pred)*100, 2)) + ' %.')


print('Accuracy of my KNN model is equal ' + str(round(accuracy_score(y_test, pred)*100, 2)) + ' %.')


# In[137]:


from sklearn import metrics

print('Performance metrix for sklearn KNN classifer')
print(metrics.classification_report(y_test, pred,
target_names=['setosa','versicolor','virginica']) )

print('Performance metrix for my KNN classifer')
print(metrics.classification_report(y_test, mypred,
target_names=['setosa','versicolor','virginica']) )

input('Press any [key] to exit!')