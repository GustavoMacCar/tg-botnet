import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from models.knn_model import knn
from models.rf_model import random_forest
from models.ada_boost_model import ada_boost

#random forest
#naive bayes
#ada boost
#CNN
#decision tree
#svm

# 1 no mesmo dataset (Domínio simples)
# 2 entre datasets diferentes (Domínio misto/adaptado)
# 3 misturar datasets (Domínio expandido)

#auc
#f1 score
#recall
#acurácia
#precision

features = pd.read_csv('ctu_13_4.csv')
labels = np.array(features['Botnet'])

features = features.drop('Botnet', axis = 1)

#l = [i for i in range(77) if i > 57]
l = [4, 6, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 25, 26, 30, 31, 32, 36, 37, 39]
r = [i for i in range(77) if i not in l]

features = features.drop(features.columns[r], axis=1)
#print(features)
feature_list = list(features.columns)
features = np.array(features)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

knn(train_features, train_labels, test_features, test_labels, 5)
print()

random_forest(train_features, train_labels, test_features, test_labels)
print()

ada_boost(train_features, train_labels, test_features, test_labels)
print()
