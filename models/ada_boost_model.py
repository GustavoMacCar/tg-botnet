from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score, auc, accuracy_score, precision_score, recall_score
from utility import is_csv_empty
import time
import csv

def ada_boost(train_features, train_labels, test_features, test_labels, result, dataset):
    ada = AdaBoostClassifier(n_estimators=100)

    start = time.time()
    ada.fit(train_features, train_labels)

    ada_predictions = ada.predict(test_features)
    end = time.time()
    
    ada_errors = abs(ada_predictions - test_labels)
    
    ada_f1 = f1_score(test_labels, ada_predictions)
    ada_accuracy = accuracy_score(test_labels, ada_predictions)
    ada_precision = precision_score(test_labels, ada_predictions)
    ada_recall = recall_score(test_labels, ada_predictions)
    
    # print(f"ada boost f1: {ada_f1}")
    # print(f"ada boost accuracy: {ada_accuracy}")
    # print(f"ada boost precision: {ada_precision}")
    # print(f"ada boost recall: {ada_recall}")
    # print(f"ada time for training and classification: {end - start}")

    optimizer1 = ''
    optimizer2 = ''

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


    
    
    with open(f"results/ada_boost/{dataset}/{result}.csv", mode='a+') as result_file:
        result_file = csv.writer(result_file, delimiter=',')
        result_file.writerow([f"{ada_f1}", f"{ada_accuracy}", f"{ada_precision}", f"{ada_recall}", f"{end - start}", f"{optimizer1}", f"{optimizer2}"])


