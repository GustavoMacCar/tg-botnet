import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from models.knn_model import knn
from models.rf_model import random_forest
from models.ada_boost_model import ada_boost
from models.decision_tree_model import decision_tree
from models.naive_bayes_model import bernoulli
from models.svm_model import svm
import random
import sys

#knn ok
#random forest ok
#naive bayes (bernouli, por tratar-se de calssificação binária -> botnet ou não) ok
#ada boost ok
#CNN
#decision tree ok
#svm ok

# 1 no mesmo dataset (Domínio simples)
# 2 entre datasets diferentes (Domínio misto/adaptado)
# 3 misturar datasets (Domínio expandido)

#auc
#f1 score
#recall
#acurácia
#precision
features = pd.read_csv('ctu13_3.csv')
labels = np.array(features['Botnet'])

features = features.drop('Botnet', axis = 1)

#l = [i for i in range(77) if i > 57]
#l = [4, 6, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 25, 26, 30, 31, 32, 36, 37, 39]

optimized_features = []
with open('features.csv') as csv_file:
    next(csv_file)
    for row in csv.reader(csv_file):
        row = [int(i) for i in row]
        optimized_features.append(row)
optimized_features = [e for e in optimized_features if e != []]

if len(optimized_features) > 0:
    for optimized_feature in optimized_features:
        features = pd.DataFrame(features)
        mocked_features = features
        #r = [i for i in range(77) if i not in optimized_feature]
        
        mocked_features = features.iloc[:, [i for i in range(22)]]
        feature_list = list(mocked_features.columns)
        mocked_features = np.array(mocked_features)

       
        train_features, test_features, train_labels, test_labels = train_test_split(mocked_features, labels, test_size = 0.25, random_state = int(random.random()*100000))
        knn(train_features, train_labels, test_features, test_labels, 5, sys.argv[1])
        #print()
        
        # random_forest(train_features, train_labels, test_features, test_labels)
        # print()
        
        # ada_boost(train_features, train_labels, test_features, test_labels)
        # print()

        # decision_tree(train_features, train_labels, test_features, test_labels)
        # print()

        # bernoulli(train_features, train_labels, test_features, test_labels)
        # print()
        
        # svm(train_features, train_labels, test_features, test_labels)
        # print()
        