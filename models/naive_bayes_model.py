from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import f1_score, auc, accuracy_score, precision_score, recall_score
import time

def bernoulli(train_features, train_labels, test_features, test_labels, result):
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
    
    print(f"bernoulli f1: {bernoulli_f1}")
    print(f"bernoulli accuracy: {bernoulli_accuracy}")
    print(f"bernoulli precision: {bernoulli_precision}")
    print(f"bernoulli recall: {bernoulli_recall}")
    print(f"bernoulli time for training and classification: {end - start}")

    