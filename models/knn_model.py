from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, auc, accuracy_score, precision_score, recall_score

def knn(train_features, train_labels, test_features, test_labels, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors)
    
    knn.fit(train_features, train_labels)
    
    knn_predictions = knn.predict(test_features)
    
    knn_errors = abs(knn_predictions - test_labels)
    
    knn_f1 = f1_score(test_labels, knn_predictions)
    knn_accuracy = accuracy_score(test_labels, knn_predictions)
    knn_precision = precision_score(test_labels, knn_predictions)
    knn_recall = recall_score(test_labels, knn_predictions)
    
    print(f"knn f1: {knn_f1}")
    print(f"knn accuracy: {knn_accuracy}")
    print(f"knn precision: {knn_precision}")
    print(f"knn recall: {knn_recall}")
    
    
    
    















