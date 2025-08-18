import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, RocCurveDisplay)
from sklearn.ensemble import RandomForestClassifier

url = (r"D:\Kartik\Learning\ML\Data\creditcard.csv")
df = pd.read_csv(url)

print(df.head())
print(df.info())
print(df['Class'].value_counts())  # For value count of each type

true = df[df.Class == 0]
false = df[df.Class == 1]
# to Check - Shape, description, details

true_sample = true.sample(n=492, random_state=42)
new_df = pd.concat([true_sample, false], axis = 0)

print(new_df.head())
print(new_df['Class'].value_counts())

scaler = StandardScaler()
x = new_df.drop(['Time', 'Class'], axis = 1)
X = scaler.fit_transform(x)
y = new_df['Class']

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state = 42, test_size = 0.2, stratify=y)

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(Xtrain,ytrain)

ypred = model.predict(Xtest)
y_pred_proba = model.predict_proba(Xtest)[:, 1]

acc = accuracy_score(ytest, ypred)
prec = precision_score(ytest, ypred)
rec = recall_score(ytest, ypred)
f1 = f1_score(ytest, ypred)
roc_auc = roc_auc_score(ytest, y_pred_proba)

print("\nModel Performance:")
print(f"Accuracy     : {acc:.4f}")
print(f"Precision    : {prec:.4f}")
print(f"Recall       : {rec:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
print(confusion_matrix(ytest, ypred))

cm = confusion_matrix(ytest, ypred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

RocCurveDisplay.from_estimator(model, Xtest, ytest)
plt.show()

