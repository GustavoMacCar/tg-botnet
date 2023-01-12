from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score, auc, accuracy_score, precision_score, recall_score
import time

def ada_boost(train_features, train_labels, test_features, test_labels):
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
    
    print(f"ada boost f1: {ada_f1}")
    print(f"ada boost accuracy: {ada_accuracy}")
    print(f"ada boost precision: {ada_precision}")
    print(f"ada boost recall: {ada_recall}")
    print(f"ada time for training and classification: {end - start}")
