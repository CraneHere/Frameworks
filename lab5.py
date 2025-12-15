#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from sklearn.metrics import (
accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
mean_absolute_error, mean_squared_error, r2_score
)

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# путь к файлу (предполагается, что CSV уже скачан)
data_cls = pd.read_csv('UCI_Credit_Card.csv')

data_cls.head()


# In[3]:


# Целевая переменная
target = 'default.payment.next.month'

X = data_cls.drop(columns=[target, 'ID'])
y = data_cls[target]

# Деление на train / test
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y
)


# In[4]:


clf = GradientBoostingClassifier(
n_estimators=200,
learning_rate=0.05,
max_depth=3,
random_state=42
)

clf.fit(X_train, y_train)


# In[5]:


y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

accuracy, precision, recall, f1, roc_auc


# In[6]:


data_reg = pd.read_csv('AirQuality.csv', sep=';', decimal=',')

data_reg.head()


# In[7]:


# Удаляем пустые столбцы
empty_cols = data_reg.columns[data_reg.isna().all()]
data_reg = data_reg.drop(columns=empty_cols)

# Удаляем строки с пропусками
data_reg = data_reg.replace(-200, np.nan)
data_reg = data_reg.dropna()

# Целевая переменная (пример: CO)
target_reg = 'CO(GT)'

X = data_reg.drop(columns=[target_reg, 'Date', 'Time'])
y = data_reg[target_reg]

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)


# In[8]:


reg = GradientBoostingRegressor(
n_estimators=300,
learning_rate=0.05,
max_depth=3,
random_state=42
)

reg.fit(X_train, y_train)


# In[9]:


y_pred = reg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

mae, mse, rmse, r2


# In[10]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from sklearn.metrics import (
accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
mean_absolute_error, mean_squared_error, r2_score
)

import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data_cls = pd.read_csv('UCI_Credit_Card.csv')

data_cls.head()


# In[ ]:


target = 'default.payment.next.month'

X = data_cls.drop(columns=[target, 'ID'])
y = data_cls[target]

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y
)


# In[13]:


clf = GradientBoostingClassifier(
n_estimators=200,
learning_rate=0.05,
max_depth=3,
random_state=42
)

clf.fit(X_train, y_train)


# In[14]:


y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

accuracy, precision, recall, f1, roc_auc


# In[15]:


data_reg = pd.read_csv('AirQuality.csv', sep=';', decimal=',')

data_reg.head()


# In[ ]:


empty_cols = data_reg.columns[data_reg.isna().all()]
data_reg = data_reg.drop(columns=empty_cols)

data_reg = data_reg.replace(-200, np.nan)
data_reg = data_reg.dropna()

target_reg = 'CO(GT)'

X = data_reg.drop(columns=[target_reg, 'Date', 'Time'])
y = data_reg[target_reg]

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)


# In[17]:


reg = GradientBoostingRegressor(
n_estimators=300,
learning_rate=0.05,
max_depth=3,
random_state=42
)

reg.fit(X_train, y_train)


# In[18]:


y_pred = reg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

mae, mse, rmse, r2


# In[ ]:


data_cls = pd.read_csv('UCI_Credit_Card.csv')
data_cls = data_cls.drop(columns=['ID'])

X = data_cls.drop(columns=['default.payment.next.month'])
y = data_cls['default.payment.next.month']

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[20]:


from sklearn.model_selection import GridSearchCV

param_grid = {
'n_estimators': [100, 200],
'learning_rate': [0.05, 0.1],
'max_depth': [2, 3]
}

grid_clf = GridSearchCV(
GradientBoostingClassifier(random_state=42),
param_grid,
scoring='f1',
cv=5,
n_jobs=-1
)

grid_clf.fit(X_train_scaled, y_train)

grid_clf.best_params_


# In[21]:


best_clf = grid_clf.best_estimator_
best_clf.fit(X_train_scaled, y_train)

y_pred = best_clf.predict(X_test_scaled)
y_proba = best_clf.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

accuracy, precision, recall, f1, roc_auc


# In[ ]:


data_reg = pd.read_csv('AirQuality.csv', sep=';', decimal=',')

data_reg = data_reg.dropna(axis=1, how='all')

data_reg = data_reg.replace(-200, np.nan).dropna()

target_reg = 'CO(GT)'

X = data_reg.drop(columns=[target_reg, 'Date', 'Time'])
y = data_reg[target_reg]

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[23]:


param_grid = {
'n_estimators': [200, 300],
'learning_rate': [0.05, 0.1],
'max_depth': [2, 3]
}

grid_reg = GridSearchCV(
GradientBoostingRegressor(random_state=42),
param_grid,
scoring='neg_mean_squared_error',
cv=5,
n_jobs=-1
)

grid_reg.fit(X_train_scaled, y_train)

grid_reg.best_params_


# In[24]:


best_reg = grid_reg.best_estimator_
best_reg.fit(X_train_scaled, y_train)

y_pred = best_reg.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

mae, rmse, r2


# In[25]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from sklearn.metrics import (
accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
mean_absolute_error, mean_squared_error, r2_score
)

import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data_cls = pd.read_csv('UCI_Credit_Card.csv')

data_cls.head()


# In[27]:


# Целевая переменная
target = 'default.payment.next.month'

X = data_cls.drop(columns=[target, 'ID'])
y = data_cls[target]

# Деление на train / test
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y
)


# In[28]:


clf = GradientBoostingClassifier(
n_estimators=200,
learning_rate=0.05,
max_depth=3,
random_state=42
)

clf.fit(X_train, y_train)


# In[29]:


y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

accuracy, precision, recall, f1, roc_auc


# In[30]:


data_reg = pd.read_csv('AirQuality.csv', sep=';', decimal=',')

data_reg.head()


# In[ ]:


empty_cols = data_reg.columns[data_reg.isna().all()]
data_reg = data_reg.drop(columns=empty_cols)

data_reg = data_reg.replace(-200, np.nan)
data_reg = data_reg.dropna()

target_reg = 'CO(GT)'

X = data_reg.drop(columns=[target_reg, 'Date', 'Time'])
y = data_reg[target_reg]

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)


# In[32]:


reg = GradientBoostingRegressor(
n_estimators=300,
learning_rate=0.05,
max_depth=3,
random_state=42
)

reg.fit(X_train, y_train)


# In[33]:


y_pred = reg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

mae, mse, rmse, r2


# In[34]:


# Повторная загрузка и очистка
data_cls = pd.read_csv('UCI_Credit_Card.csv')
data_cls = data_cls.drop(columns=['ID'])

X = data_cls.drop(columns=['default.payment.next.month'])
y = data_cls['default.payment.next.month']

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [2, 3]
}

grid_clf = GridSearchCV(
GradientBoostingClassifier(random_state=42),
param_grid,
scoring='f1',
cv=5,
n_jobs=-1
)

grid_clf.fit(X_train_scaled, y_train)

grid_clf.best_params_


# In[36]:


best_clf = grid_clf.best_estimator_
best_clf.fit(X_train_scaled, y_train)

y_pred = best_clf.predict(X_test_scaled)
y_proba = best_clf.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

accuracy, precision, recall, f1, roc_auc


# In[ ]:


data_reg = pd.read_csv('AirQuality.csv', sep=';', decimal=',')

data_reg = data_reg.dropna(axis=1, how='all')

data_reg = data_reg.replace(-200, np.nan).dropna()

target_reg = 'CO(GT)'

X = data_reg.drop(columns=[target_reg, 'Date', 'Time'])
y = data_reg[target_reg]

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[38]:


param_grid = {
'n_estimators': [200, 300],
'learning_rate': [0.05, 0.1],
'max_depth': [2, 3]
}

grid_reg = GridSearchCV(
GradientBoostingRegressor(random_state=42),
param_grid,
scoring='neg_mean_squared_error',
cv=5,
n_jobs=-1
)

grid_reg.fit(X_train_scaled, y_train)

grid_reg.best_params_


# In[39]:


best_reg = grid_reg.best_estimator_
best_reg.fit(X_train_scaled, y_train)

y_pred = best_reg.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

mae, rmse, r2


# In[ ]:


data_cls = pd.read_csv('UCI_Credit_Card.csv')

X_cls = data_cls.drop(columns=['ID', 'default.payment.next.month'])
y_cls = data_cls['default.payment.next.month'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X_cls, y_cls,
    test_size=0.2,
    random_state=42,
    stratify=y_cls
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[ ]:


data_air = pd.read_csv(
    'AirQuality.csv',
    sep=';',
    decimal=','
)

data_air = data_air.dropna(axis=1, how='all')

data_air = data_air.replace(-200, np.nan).dropna()

target_air = 'CO(GT)'

X_air = data_air.drop(columns=[target_air, 'Date', 'Time'])
y_air = data_air[target_air].astype(float)

X_air_train, X_air_test, y_air_train, y_air_test = train_test_split(
    X_air, y_air,
    test_size=0.2,
    random_state=42
)

scaler_air = StandardScaler()
X_air_train_scaled = scaler_air.fit_transform(X_air_train)
X_air_test_scaled = scaler_air.transform(X_air_test)


# In[60]:


class LogisticRegressionGD:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.w = None
        self.b = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iter):
            linear = np.dot(X, self.w) + self.b
            y_pred = self.sigmoid(linear)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        linear = np.dot(X, self.w) + self.b
        probs = self.sigmoid(linear)
        return (probs >= 0.5).astype(int)


# In[61]:


class LinearRegressionGD:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iter):
            y_pred = np.dot(X, self.w) + self.b

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b


# In[62]:


print("y_test:")
print(type(y_test), y_test.shape, y_test.dtype)
print(np.unique(y_test)[:10])

print("\ny_pred:")
print(type(y_pred), y_pred.shape, y_pred.dtype)
print(np.unique(y_pred)[:10])


# In[ ]:


log_reg = LogisticRegressionGD(lr=0.1, n_iter=2000)
log_reg.fit(X_train_scaled, y_train)

y_pred = log_reg.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

accuracy, precision, recall, f1


# In[64]:


lin_reg = LinearRegressionGD(lr=0.01, n_iter=3000)
lin_reg.fit(X_train_scaled, y_train)

y_pred = lin_reg.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

mae, rmse, r2


# In[65]:


log_reg = LogisticRegressionGD(lr=0.05, n_iter=4000)
log_reg.fit(X_train_scaled, y_train)

y_pred = log_reg.predict(X_test_scaled)

f1 = f1_score(y_test, y_pred)
f1


# In[66]:


lin_reg = LinearRegressionGD(lr=0.005, n_iter=5000)
lin_reg.fit(X_train_scaled, y_train)

y_pred = lin_reg.predict(X_test_scaled)

rmse = mean_squared_error(y_test, y_pred, squared=False)
rmse

