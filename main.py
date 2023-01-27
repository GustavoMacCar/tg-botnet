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

SAME_DATASET = True

features = pd.read_csv('ctu13_sample.csv')
labels = np.array(features['Botnet'])
features = features.drop('Botnet', axis = 1)

if not SAME_DATASET:
    test_dataset = pd.read_csv('ctu13_sample.csv')
    test_labels = np.array(test_dataset['Botnet'])
    #test_dataset = test_dataset.drop(test_dataset['Botnet'])
    del test_dataset['Botnet']

optimized_features = []

if not is_csv_empty('features.csv'):
    with open('features.csv') as csv_file:
        next(csv_file)
        for row in csv.reader(csv_file):
            row = [int(i) for i in row]
            optimized_features.append(row)
    optimized_features = [e for e in optimized_features if e != []]

if len(optimized_features) > 0:
    if SAME_DATASET:
        for optimized_feature in optimized_features: #always has only one element: an array with the optmized features
            features = pd.DataFrame(features)
            selected_features = features.iloc[:,optimized_feature].copy()
            
            selected_features = np.array(selected_features)
            
            train_features, test_features, train_labels, test_labels = train_test_split(selected_features, labels, test_size = 0.25, random_state = int(random.random()*100000))
            #knn(train_features, train_labels, test_features, test_labels, 5, sys.argv[1], sys.argv[2])
            #print()
            
            #random_forest(train_features, train_labels, test_features, test_labels, sys.argv[1], sys.argv[2])
            # print()
            
            #ada_boost(train_features, train_labels, test_features, test_labels, sys.argv[1], sys.argv[2])
            # print()
    
            decision_tree(train_features, train_labels, test_features, test_labels, sys.argv[1], sys.argv[2])
            # print()
    
            #bernoulli(train_features, train_labels, test_features, test_labels, sys.argv[1], sys.argv[2])
            # print()
            
            #svm(train_features, train_labels, test_features, test_labels, sys.argv[1], sys.argv[2])
            # print()
    
    else: #cross datasets
        for optimized_feature in optimized_features: #always has only one element: an array with the optmized features
            features = pd.DataFrame(features)

            selected_features = features.iloc[:,optimized_feature].copy()
            selected_features = np.array(selected_features)
            
            test_features = test_dataset.iloc[:,optimized_feature].copy()
            test_features = np.array(test_features)

            #knn(selected_features, labels, test_features, test_labels, 5, sys.argv[1], sys.argv[2])
            #print()
            
            #random_forest(selected_features, labels, test_features, test_labels, sys.argv[1], sys.argv[2])
            # print()
            
            #ada_boost(selected_features, labels, test_features, test_labels, sys.argv[1], sys.argv[2])
            # print()
    
            decision_tree(selected_features, labels, test_features, test_labels, sys.argv[1], sys.argv[2])
            # print()
    
            #bernoulli(selected_features, labels, test_features, test_labels, sys.argv[1], sys.argv[2])
            # print()
            
            #svm(selected_features, labels, test_features, test_labels, sys.argv[1], sys.argv[2])
            # print()


else: #control
    features = pd.DataFrame(features)
    selected_features = features
    
    selected_features = np.array(selected_features)
    for i in range(30):
        train_features, test_features, train_labels, test_labels = train_test_split(selected_features, labels, test_size = 0.25, random_state = int(random.random()*100000))
    
        #knn(train_features, train_labels, test_features, test_labels, 5, 'control', sys.argv[2])
        #print()
        
        #random_forest(train_features, train_labels, test_features, test_labels, 'control', sys.argv[2])
        # print()
        
        #ada_boost(train_features, train_labels, test_features, test_labels, 'control', sys.argv[2])
        # print()
    
        decision_tree(train_features, train_labels, test_features, test_labels, 'control', sys.argv[2])
        # print()
    
        #bernoulli(train_features, train_labels, test_features, test_labels, 'control', sys.argv[2])
        # print()
        
        #svm(train_features, train_labels, test_features, test_labels, 'control', sys.argv[2])
        # print()
    
    