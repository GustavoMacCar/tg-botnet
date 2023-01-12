from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, auc, accuracy_score, precision_score, recall_score
import time
import random

def decision_tree(train_features, train_labels, test_features, test_labels):
    decision_tree = DecisionTreeClassifier(max_leaf_nodes=10)
    
    start = time.time()
    decision_tree.fit(train_features, train_labels)
    decision_tree_predictions = decision_tree.predict(test_features)
    end = time.time()
    
    decision_tree_errors = abs(decision_tree_predictions - test_labels)
    
    decision_tree_f1 = f1_score(test_labels, decision_tree_predictions)
    decision_tree_accuracy = accuracy_score(test_labels, decision_tree_predictions)
    decision_tree_precision = precision_score(test_labels, decision_tree_predictions)
    decision_tree_recall = recall_score(test_labels, decision_tree_predictions)
    
    print(f"decision_tree f1: {decision_tree_f1}")
    print(f"decision_tree accuracy: {decision_tree_accuracy}")
    print(f"decision_tree precision: {decision_tree_precision}")
    print(f"decision_tree recall: {decision_tree_recall}")
    print(f"decision_tree time for training and classification: {end - start}")

    
    
    