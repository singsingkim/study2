import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, AveragePooling1D, Flatten, Conv2D, LSTM, Bidirectional, Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt

path = "c:/_data/kaggle/비만/"
train = pd.read_csv(path + "train.csv", index_col=0)
test = pd.read_csv(path + "test.csv", index_col=0)
sample = pd.read_csv(path + "sample_submission.csv")
x = train.drop(['NObeyesdad'], axis=1)
y = train['NObeyesdad']

# Label Encoding
lb = LabelEncoder()
columns_to_encode = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
for column in columns_to_encode:
    lb.fit(x[column])
    x[column] = lb.transform(x[column])
    lb.fit(test[column])
    test[column] = lb.transform(test[column])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=21, stratify=y)

lgbm_params = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "random_state": 21,
    "num_class": 7,
    "learning_rate": 0.01386432121252535,
    'n_estimators': 1000,
    'feature_pre_filter': False,
    'lambda_l1': 1.2149501037669967e-07,
    'lambda_l2': 0.9230890143196759,
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5523862448863431,
    'bagging_freq': 4,
    'min_child_samples': 20
}

model = LGBMClassifier(**lgbm_params)
model.fit(x_train, y_train)

# Outlier Detection Function
def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out > upper_bound) | (data_out < lower_bound))

# Detect outliers in a feature
feature_index = 0  # Index of the feature you want to detect outliers for
feature_name = x.columns[feature_index]
outliers_loc = outliers(x[feature_name])
print('Outliers in', feature_name, 'at positions:', outliers_loc)

# Boxplot Visualization
plt.boxplot(x[feature_name])
plt.title('Boxplot of ' + feature_name)
plt.show()

# Model Evaluation
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
