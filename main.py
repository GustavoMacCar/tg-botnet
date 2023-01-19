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
from utility import is_csv_empty
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

if not is_csv_empty('features.csv'):
    with open('features.csv') as csv_file:
        next(csv_file)
        for row in csv.reader(csv_file):
            row = [int(i) for i in row]
            optimized_features.append(row)
    optimized_features = [e for e in optimized_features if e != []]

if len(optimized_features) > 0:
    for optimized_feature in optimized_features: #always has only one element: an array with the optmized features
        features = pd.DataFrame(features)
        selected_features = features.iloc[:,optimized_feature].copy()
        
        selected_features = np.array(selected_features)
        
        train_features, test_features, train_labels, test_labels = train_test_split(selected_features, labels, test_size = 0.25, random_state = int(random.random()*100000))
        knn(train_features, train_labels, test_features, test_labels, 5, sys.argv[1], sys.argv[2])
        #print()
        
        #random_forest(train_features, train_labels, test_features, test_labels, sys.argv[1], sys.argv[2])
        # print()
        
        #ada_boost(train_features, train_labels, test_features, test_labels, sys.argv[1], sys.argv[2])
        # print()

        #decision_tree(train_features, train_labels, test_features, test_labels, sys.argv[1], sys.argv[2])
        # print()

        #bernoulli(train_features, train_labels, test_features, test_labels, sys.argv[1], sys.argv[2])
        # print()
        
        #svm(train_features, train_labels, test_features, test_labels, sys.argv[1], sys.argv[2])
        # print()

else:
    features = pd.DataFrame(features)
    selected_features = features
    #r = [i for i in range(77) if i not in optimized_feature]
    
    selected_features = np.array(selected_features)
    for i in range(30):
        train_features, test_features, train_labels, test_labels = train_test_split(selected_features, labels, test_size = 0.25, random_state = int(random.random()*100000))
    
        #knn(train_features, train_labels, test_features, test_labels, 5, 'control', sys.argv[2])
        #print()
        
        random_forest(train_features, train_labels, test_features, test_labels, 'control', sys.argv[2])
        # print()
        
        #ada_boost(train_features, train_labels, test_features, test_labels, 'control', sys.argv[2])
        # print()
    
        #decision_tree(train_features, train_labels, test_features, test_labels, 'control', sys.argv[2])
        # print()
    
        #bernoulli(train_features, train_labels, test_features, test_labels, 'control', sys.argv[2])
        # print()
        
        #svm(train_features, train_labels, test_features, test_labels, 'control', sys.argv[2])
        # print()
    
    