import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
from google.colab import files
uploaded = files.upload()
df = pd.read_csv('parkinsons.data')
df.head()
df.isnull().values.any()
df.shape

df['status'].value_counts()

percent_has_disease = 147 / (147 + 48) * 100
percent_has_not_disease = 48 / (147 + 48) * 100

print("percent_has_disease" , percent_has_disease)
print("percent_has_not_disease" , percent_has_not_disease)

X = df.drop(['name'], 1)
X = np.array(X.drop(['status'],1))
y = np.array(df['status'])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

model = XGBClassifier().fit(x_train, y_train)

predictions = model.predict(x_test)
print(predictions)

print(y_test)

print(classification_report(y_test, predictions))
  