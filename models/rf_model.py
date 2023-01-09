from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, auc, accuracy_score, precision_score, recall_score


def random_forest(train_features, train_labels, test_features, test_labels):
    rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
    
    rf.fit(train_features, train_labels)
    
    rf_predictions = rf.predict(test_features)
    
    rf_errors = abs(rf_predictions - test_labels)
    
    rf_f1 = f1_score(test_labels, rf_predictions)
    rf_accuracy = accuracy_score(test_labels, rf_predictions)
    rf_precision = precision_score(test_labels, rf_predictions)
    rf_recall = recall_score(test_labels, rf_predictions)
    
    print(f"rf f1: {rf_f1}")
    print(f"rf accuracy: {rf_accuracy}")
    print(f"rf precision: {rf_precision}")
    print(f"rf recall: {rf_recall}")
    