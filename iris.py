import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"A:\practice_files\Iris.csv")
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
df = df.drop_duplicates()
print(df.duplicated().sum())

x = df.drop("Species", axis=1)
y = df["Species"]

df["Species"] = LabelEncoder().fit_transform(df["Species"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

reg = LogisticRegression().fit(x_train, y_train)
knn = KNeighborsClassifier().fit(x_train, y_train)
svm = SVC().fit(x_train, y_train)

pred1 = reg.predict(x_test)
pred2 = knn.predict(x_test)
pred3 = svm.predict(x_test)

print(f"LR: %{accuracy_score(y_test, pred1) * 100}")
print(f"KNN: %{accuracy_score(y_test, pred2) * 100}")
print(f"SVM: %{accuracy_score(y_test, pred3) * 100}")

import matplotlib.pyplot as plt
import seaborn as sns
plt.subplots(figsize = (6, 6))
sns.heatmap(df.corr(), annot=True)
plt.show()