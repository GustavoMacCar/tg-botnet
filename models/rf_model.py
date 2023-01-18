from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, auc, accuracy_score, precision_score, recall_score
from utility import is_csv_empty
import time
import csv

def random_forest(train_features, train_labels, test_features, test_labels, result, dataset):
    rf = RandomForestClassifier(n_estimators = 1000)
    
    start = time.time()
    rf.fit(train_features, train_labels)
    
    rf_predictions = rf.predict(test_features)
    end = time.time()
    
    rf_errors = abs(rf_predictions - test_labels)
    
    rf_f1 = f1_score(test_labels, rf_predictions)
    rf_accuracy = accuracy_score(test_labels, rf_predictions)
    rf_precision = precision_score(test_labels, rf_predictions)
    rf_recall = recall_score(test_labels, rf_predictions)
    
    # print(f"rf f1: {rf_f1}")
    # print(f"rf accuracy: {rf_accuracy}")
    # print(f"rf precision: {rf_precision}")
    # print(f"rf recall: {rf_recall}")
    # print(f"rf time for training and classification: {end - start}")

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


    
    
    with open(f"/home/gmcma/tg/tg-botnet/results/rf/{dataset}/{result}.csv", mode='a') as result_file:
        result_file = csv.writer(result_file, delimiter=',')
        result_file.writerow([f"{rf_f1}", f"{rf_accuracy}", f"{rf_precision}", f"{rf_recall}", f"{end - start}", f"{optimizer1}", f"{optimizer2}"])


    