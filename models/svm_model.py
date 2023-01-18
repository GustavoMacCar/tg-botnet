from sklearn.svm import SVC
from sklearn.metrics import f1_score, auc, accuracy_score, precision_score, recall_score
from utility import is_csv_empty
import time
import csv

def svm(train_features, train_labels, test_features, test_labels, result, dataset):
    svm = SVC()

    start = time.time()
    svm.fit(train_features, train_labels)
    svm_predictions = svm.predict(test_features)
    end = time.time()
    
    svm_errors = abs(svm_predictions - test_labels)
    
    svm_f1 = f1_score(test_labels, svm_predictions)
    svm_accuracy = accuracy_score(test_labels, svm_predictions)
    svm_precision = precision_score(test_labels, svm_predictions)
    svm_recall = recall_score(test_labels, svm_predictions)
    
    # print(f"svm f1: {svm_f1}")
    # print(f"svm accuracy: {svm_accuracy}")
    # print(f"svm precision: {svm_precision}")
    # print(f"svm recall: {svm_recall}")
    # print(f"svm time for training and classification: {end - start}")

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


    
    
    with open(f"/home/gmcma/tg/tg-botnet/results/svm/{dataset}/{result}.csv", mode='a') as result_file:
        result_file = csv.writer(result_file, delimiter=',')
        result_file.writerow([f"{svm_f1}", f"{svm_accuracy}", f"{svm_precision}", f"{svm_recall}", f"{end - start}", f"{optimizer1}", f"{optimizer2}"])


    