import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, auc, accuracy_score, precision_score, recall_score

#random forest
#naive bayes
#ada boost
#CNN
#decision tree
#svm

# 1 no mesmo dataset (Domínio simples)
# 2 entre datasets diferentes (Domínio misto/adaptado)
# 3 misturar datasets (Domínio expandido)

#auc
#f1 score
#recall
#acurácia
#precision

features = pd.read_csv('ctu13_3.csv')
labels = np.array(features['Botnet'])

features = features.drop('Botnet', axis = 1)

# features = features.drop(features.columns[0:28], axis=1)

feature_list = list(features.columns)
features = np.array(features)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
knn = KNeighborsClassifier(n_neighbors=5)

rf.fit(train_features, train_labels)
knn.fit(train_features, train_labels)

knn_predictions = knn.predict(test_features)
rf_predictions = rf.predict(test_features)

knn_errors = abs(knn_predictions - test_labels)
# print(round(np.mean(knn_errors), 2))

rf_errors = abs(rf_predictions - test_labels)
# print(round(np.mean(rf_errors), 2))

knn_f1 = f1_score(test_labels, knn_predictions)
knn_accuracy = accuracy_score(test_labels, knn_predictions)
knn_precision = precision_score(test_labels, knn_predictions)
knn_recall = recall_score(test_labels, knn_predictions)

rf_f1 = f1_score(test_labels, rf_predictions)
rf_accuracy = accuracy_score(test_labels, rf_predictions)
rf_precision = precision_score(test_labels, rf_predictions)
rf_recall = recall_score(test_labels, rf_predictions)

print(f"knn f1: {knn_f1}")
print(f"knn accuracy: {knn_accuracy}")
print(f"knn precision: {knn_precision}")
print(f"knn recall: {knn_recall}")

print("\n")

print(f"rf f1: {rf_f1}")
print(f"rf accuracy: {rf_accuracy}")
print(f"rf precision: {rf_precision}")
print(f"rf recall: {rf_recall}")





















