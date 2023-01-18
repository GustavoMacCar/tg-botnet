from sklearn.svm import SVC
from sklearn.metrics import f1_score, auc, accuracy_score, precision_score, recall_score
import time

def svm(train_features, train_labels, test_features, test_labels, result):
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
    
    print(f"svm f1: {svm_f1}")
    print(f"svm accuracy: {svm_accuracy}")
    print(f"svm precision: {svm_precision}")
    print(f"svm recall: {svm_recall}")
    print(f"svm time for training and classification: {end - start}")

    