import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('processed_data.csv')

array = data.values
X = array[:, 0:7]
Y = array[:, 7]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

# load model
print("====== Loading Model ======= \n")
rfc_model = joblib.load("rfc_model.joblib")
pred = rfc_model.predict(x_train)

# %%
# test data 
inputs =  np.expand_dims(x_test[0], 0)

print("===== Predicting ==== \n")
prediction = rfc_model.predict(inputs)

print("Predicted Target = ", np.squeeze(prediction, -1))

if prediction == 1:
    print("Student is likely to dropout")
elif prediction == 0:
    print("Student is not likely to dropout")
