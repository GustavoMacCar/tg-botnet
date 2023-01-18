from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import f1_score, auc, accuracy_score, precision_score, recall_score
import time
import csv

def bernoulli(train_features, train_labels, test_features, test_labels, result, dataset):
    bernoulli = BernoulliNB()

    start = time.time()
    bernoulli.fit(train_features, train_labels)
    bernoulli_predictions = bernoulli.predict(test_features)
    end = time.time()
    
    bernoulli_errors = abs(bernoulli_predictions - test_labels)
    
    bernoulli_f1 = f1_score(test_labels, bernoulli_predictions)
    bernoulli_accuracy = accuracy_score(test_labels, bernoulli_predictions)
    bernoulli_precision = precision_score(test_labels, bernoulli_predictions)
    bernoulli_recall = recall_score(test_labels, bernoulli_predictions)
    
    # print(f"bernoulli f1: {bernoulli_f1}")
    # print(f"bernoulli accuracy: {bernoulli_accuracy}")
    # print(f"bernoulli precision: {bernoulli_precision}")
    # print(f"bernoulli recall: {bernoulli_recall}")
    # print(f"bernoulli time for training and classification: {end - start}")

    with open('first_optimizer.csv', mode='r') as first_optimizer:
        next(first_optimizer)
        data = csv.reader(first_optimizer, delimiter=',') 
        for row in data:
            optimizer1 = (', '.join(row))
    first_optimizer.close()

    with open('second_optimizer.csv', mode='r') as second_optimizer:
        next(second_optimizer)
        data = csv.reader(second_optimizer, delimiter=',') 
        for row in data:
            optimizer2 = (', '.join(row))
    second_optimizer.close()


    
    
    with open(f"/home/gmcma/tg/tg-botnet/results/naive_bayes/{dataset}/{result}.csv", mode='a') as result_file:
        result_file = csv.writer(result_file, delimiter=',')
        result_file.writerow([f"{bernoulli_f1}", f"{bernoulli_accuracy}", f"{bernoulli_precision}", f"{bernoulli_recall}", f"{end - start}", f"{optimizer1}", f"{optimizer2}"])


    