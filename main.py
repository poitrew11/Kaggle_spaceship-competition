#importing libraries required for operation
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
import numpy as np

#Data preprocessing
dataset = pd.read_csv('train.csv')

encoder = OneHotEncoder(sparse_output = False)

X_one = dataset.drop(['Transported', 'PassengerId', 'Name', 'Cabin'], axis = 1)
Y = dataset['Transported']

categorical_columns = X_one.select_dtypes(include = 'object').columns

X_two = X_one.drop(categorical_columns, axis = 1)
X_two = X_two.fillna(0)
X_encoded = encoder.fit_transform(X_one[categorical_columns])
new_columns = encoder.get_feature_names_out(categorical_columns)
encoded_df = pd.DataFrame(X_encoded, columns = new_columns)
New_data = pd.concat([X_two, encoded_df], axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(New_data, Y, random_state = 42, test_size = 0.25)

#model and GridSearchCV
model = RandomForestClassifier()

param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5]
}

grid_search = GridSearchCV(
    estimator = model,
    param_grid = param_grid,
    scoring = 'accuracy',
    cv = 5,
    n_jobs = -1
)

grid_search.fit(X_train, Y_train)
model_great = grid_search.best_estimator_
print(f"Best params: {grid_search.best_params_}")

#Best params: {'max_depth': 15, 'min_samples_leaf': 5, 'min_samples_split': 5, 'n_estimators': 300}
Y_pred = model_great.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)

#accuracy_score was near 0.78

#learning_curve 
train_sizes, train_scores, val_scores = learning_curve(
    estimator = model,
    X = New_data,
    y = Y,
    train_sizes = np.linspace(0.1, 1.0, 10),
    cv = 5,
    scoring = 'accuracy',
    n_jobs = -1
)

train_scores_mean = np.mean(train_scores, axis = 1)
val_scores_mean = np.mean(val_scores, axis = 1)
train_scores_std = np.std(train_scores, axis = 1)
val_scores_std = np.std(val_scores, axis = 1)

plt.figure(figsize = (10, 6))
plt.plot(train_sizes, train_scores_mean, label = 'Training_score', color = 'blue')
plt.plot(train_sizes, val_scores_mean, label = 'Validation_score', color = 'orange')

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='blue')
plt.fill_between(train_sizes, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, alpha=0.1, color='orange')

plt.xlabel('Training examples')
plt.ylabel('accuracy')
plt.grid(True)
plt.show()

#Trying to look on F1_score and change treshold for classification
thresholds = np.linspace(0, 1, 100)
f1_scores = [f1_score(Y_test, (Y_probs > t).astype(int)) for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Best_treshold: {best_threshold}")
#result was 0.44

Y_pred_05 = (Y_probs > 0.5).astype(int)
print("Confusion_matrix (threshold 0.5):\n", confusion_matrix(Y_test, Y_pred_05))

Y_pred_044 = (Y_probs > 0.44).astype(int)
print("\nConfusion_matrix (threshold 0.44):\n", confusion_matrix(Y_test, Y_pred_044))
from sklearn.metrics import f1_score

f1_05 = f1_score(Y_test, Y_pred_05)
f1_044 = f1_score(Y_test, Y_pred_044)
print(f"F1-score (threshold 0.5): {f1_05:.3f}")
print(f"F1-score (threshold 0.44): {f1_044:.3f}")

'''
Confusion_matrix (threshold 0.5):
 [[818 264]
 [200 892]]

Confusion_matrix (threshold 0.44):
 [[788 294]
 [165 927]]
F1-score (threshold 0.5): 0.794
F1-score (threshold 0.44): 0.802
'''

#ROC- AUC
Y_probs = model_great.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(Y_test, Y_probs)

roc_auc = roc_auc_score(Y_test, Y_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC-')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

#Area under curve = 0.868
#Using 0.44 as threshold - right, but this does not have a strong effect on the result.
