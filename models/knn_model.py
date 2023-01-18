from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, auc, accuracy_score, precision_score, recall_score
from utility import is_csv_empty
import time
import csv


def knn(train_features, train_labels, test_features, test_labels, n_neighbors, result, dataset):
    knn = KNeighborsClassifier(n_neighbors)
    
    start = time.time()
    knn.fit(train_features, train_labels)
    knn_predictions = knn.predict(test_features)
    end = time.time()
    
    knn_errors = abs(knn_predictions - test_labels)
    
    knn_f1 = f1_score(test_labels, knn_predictions)
    knn_accuracy = accuracy_score(test_labels, knn_predictions)
    knn_precision = precision_score(test_labels, knn_predictions)
    knn_recall = recall_score(test_labels, knn_predictions)
    
    # print(f"knn f1: {knn_f1}")
    # print(f"knn accuracy: {knn_accuracy}")
    # print(f"knn precision: {knn_precision}")
    # print(f"knn recall: {knn_recall}")
    # print(f"knn time for training and classification: {end - start}")

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


    
    
    with open(f"/home/gmcma/tg/tg-botnet/results/knn/{dataset}/{result}.csv", mode='a') as result_file:
        result_file = csv.writer(result_file, delimiter=',')
        result_file.writerow([f"{knn_f1}", f"{knn_accuracy}", f"{knn_precision}", f"{knn_recall}", f"{end - start}", f"{optimizer1}", f"{optimizer2}"])















