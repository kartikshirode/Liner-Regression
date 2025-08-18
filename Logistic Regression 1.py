import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


df = pd.read_csv(r"D:\Kartik\Learning\ML\Data\hypertension_dataset.csv")

# print(df.head())
# print(df.columns)
# print(df.shape)
# print(df.info())

df['Family_History'] = df['Family_History'].map({'Yes': 3, 'No': 1})
df['Smoking_Status'] = df['Smoking_Status'].map({'Smoker': 1, 'Non-Smoker': 0})
df['Exercise_Level'] = df['Exercise_Level'].map({'Low': 3, 'Moderate': 2, 'High': 1})
df['BP_History'] = df['BP_History'].map({'Normal': 1, 'Prehypertension': 2, 'Hypertension': 3})
df['Has_Hypertension'] = df['Has_Hypertension'].map({'Yes': 1, 'No': 0})

# df = pd.get_dummies(df, columns=['Medication'], prefix='Med', dummy_na=True)


# for col in df.select_dtypes(include='object'):
#     print(f"{col} has = {df[col].unique()}")

X = df.drop(['BMI', 'Has_Hypertension', 'Medication'], axis=1)
y = df['Has_Hypertension']

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size= 0.3, random_state=23)

model = LogisticRegression()
model.fit(Xtrain, ytrain)

ypred = model.predict(Xtest)

print("Accuracy:", accuracy_score(ytest, ypred))
print("Confusion Matrix:\n", confusion_matrix(ytest, ypred))