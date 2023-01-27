from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, auc, accuracy_score, precision_score, recall_score
from utility import is_csv_empty
import time
import csv

def decision_tree(train_features, train_labels, test_features, test_labels, result, dataset):
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
    
    # print(f"decision_tree f1: {decision_tree_f1}")
    # print(f"decision_tree accuracy: {decision_tree_accuracy}")
    # print(f"decision_tree precision: {decision_tree_precision}")
    # print(f"decision_tree recall: {decision_tree_recall}")
    # print(f"decision_tree time for training and classification: {end - start}")


    optimizer1 = ''
    optimizer2 = ''
    optimizer3 = ''

    with open('first_optimizer.csv', mode='r') as first_optimizer:
        if not is_csv_empty('first_optimizer.csv'):
            next(first_optimizer)
            data = csv.reader(first_optimizer, delimiter=',') 
            for row in data:
                optimizer1 = (', '.join(row))
    first_optimizer.close()

    with open('second_optimizer.csv', mode='r') as second_optimizer:
        if not is_csv_empty('second_optimizer.csv'):
            next(second_optimizer)
            data = csv.reader(second_optimizer, delimiter=',') 
            for row in data:
                optimizer2 = (', '.join(row))
    second_optimizer.close()

    with open('third_optimizer.csv', mode='r') as third_optimizer:
        if not is_csv_empty('third_optimizer.csv'):
            next(third_optimizer)
            data = csv.reader(third_optimizer, delimiter=',') 
            for row in data:
                optimizer3 = (', '.join(row))
    second_optimizer.close()


    
    
    with open(f"results/decision_tree/{dataset}/{result}.csv", mode='a') as result_file:
        result_file = csv.writer(result_file, delimiter=',')
        result_file.writerow([f"{decision_tree_f1}", f"{decision_tree_accuracy}", f"{decision_tree_precision}", f"{decision_tree_recall}", f"{end - start}", f"{optimizer1}", f"{optimizer2}", f"{optimizer3}"])


    
    
    