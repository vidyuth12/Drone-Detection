from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np
import pickle


data = np.loadtxt(
    'C:/Users/Vidyuth/OneDrive/Documents/Data/RF_Data.csv', delimiter=',')


x = np.transpose(data[0:2047, :])

Label_1 = np.transpose(data[2048:2049, :])
Label_1 = Label_1.astype(int)

x_train, x_test, y_train, y_test = train_test_split(
    x, Label_1, test_size=0.2, random_state=7)

model = XGBClassifier()
model.fit(x_train, y_train)

with open('xgboost_model.pkl', 'wb') as file:
    pickle.dump(model, file)

y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)

print(accuracy)
