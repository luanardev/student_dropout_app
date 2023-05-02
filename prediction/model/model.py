# import libraries
import joblib
import pandas as pd
import loaddata
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV


# LOAD DATA FRAME
processed_csv = loaddata.process('students_dataset.csv')
df = pd.read_csv(processed_csv)

# SPLIT DATA FRAME FOR MODEL FITTING
array = df.values
X = array[:, 0:7]
Y = array[:, 7]
x_train, x_validation, y_train, y_validation = train_test_split(X, Y, test_size=0.15, random_state=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

# prepare RF model
model = RandomForestClassifier()

# Make predictions on validation dataset
model.fit(x_train, y_train)
predictions = model.predict(x_validation)
confusion_matrix(y_validation, predictions)

# OverSampling with SMOTE
sm = SMOTE(random_state=42)
x_train_ovr, y_train_ovr = sm.fit_resample(x_train, y_train.ravel())

# Make predictions on validation dataset
model.fit(x_train_ovr, y_train_ovr.ravel())
predictions = model.predict(x_validation)
confusion_matrix(y_validation, predictions)

# UnderSampling with SMOTE
under_sm = RandomUnderSampler(random_state=42)
x_train_undr, y_train_undr = under_sm.fit_resample(x_train, y_train.ravel())

# Make predictions on validation dataset
model.fit(x_train_undr, y_train_undr.ravel())
predictions = model.predict(x_validation)
confusion_matrix(y_validation, predictions)
# %%

# HYPERPARAMETER OPTIMIZATION

# define search space
space = {
    'n_estimators': [115, 100],
    'criterion': ['gini', 'entropy'],
    'max_depth': range(2, 20, 1),
    'min_samples_leaf': [2, 3, 4, 5, 6],
    'min_samples_split': [3, 2, 5, 4, 6]
}
	
# define evaluation
cv = RepeatedStratifiedKFold()

# define search
search = GridSearchCV(estimator=model, param_grid=space, scoring='accuracy', cv=3, n_jobs=-1, verbose=2)

# execute search
result = search.fit(x_train_undr, y_train_undr.ravel())

# best model parameters
best = result.best_params_

print("===Best Model Parameters === \n")

# print best 
print(best)

# %%
best['max_depth'] = int(best['max_depth']) # convert to int
best["n_estimators"] = int(best['n_estimators']) # convert to int
best["min_samples_leaf"] = int(best['min_samples_leaf']) # convert to int
best["min_samples_split"] = int(best['min_samples_split']) # convert to int

# Rebuild model with optimized parameters
rfc_model = RandomForestClassifier(**best)
rfc_model.fit(x_train_undr, y_train_undr.ravel())
predictions = rfc_model.predict(x_validation)
confusion_matrix(y_validation, predictions)

# save
joblib.dump(rfc_model, "rfc_model.joblib")

print("Model training completed")
