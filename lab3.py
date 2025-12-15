#!/usr/bin/env python
# coding: utf-8

# # Лабораторная работа №3: Деревья решений и настройка гиперпараметров
# 
# ## Часть 1: Классификация с помощью дерева решений
# 
# ### Импорт необходимых библиотек

# In[70]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


# ### Загрузка и предобработка данных

# Загрузка датасета по умолчанию кредитных карт
# 

# In[71]:


df = pd.read_csv('UCI_Credit_Card.csv')
df.head()


# Проверка на пропущенные значения
# 

# In[72]:


df.info()
df.isnull().sum()


# ### Подготовка данных для моделирования

# Разделение на признаки и целевую переменную

# In[73]:


X = df.drop('default.payment.next.month', axis=1)
y = df['default.payment.next.month']


# Разделение на обучающую и тестовую выборки

# In[74]:


X_train, X_test, y_train, y_test = train_test_split(
X, y,
test_size=0.3,
random_state=42,
stratify=y
)

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")
print(f"Распределение классов в обучающей выборке: \n{y_train.value_counts(normalize=True)}")


# ### Обучение базового дерева решений

# Создание и обучение модели

# In[75]:


clf = DecisionTreeClassifier(
random_state=42
)

clf.fit(X_train, y_train)


# Предсказания

# In[76]:


y_pred = clf.predict(X_test)


# # Оценка модели

# In[77]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


# In[78]:


print("Метрики базового дерева решений:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")


# Матрица ошибок

# In[79]:


cm = confusion_matrix(y_test, y_pred)
print("Матрица ошибок:")
print(cm)


# ## Часть 2: Регрессия с помощью дерева решений
# 
# ### Загрузка и предобработка данных о качестве воздуха

# In[80]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Загрузка датасета

# In[81]:


df_air = pd.read_csv('AirQuality.csv', sep=';', decimal=',')


# Удаление полностью пустых столбцов

# In[82]:


df_air = df_air.dropna(axis=1, how='all')


# Замена значения -200 (пропуски) на NaN и удаление строк с пропусками

# In[83]:


df_air.replace(-200, np.nan, inplace=True)
df_air.dropna(inplace=True)


# ### Подготовка данных для регрессии

# Разделение на признаки и целевую переменную

# In[84]:


X = df_air.drop(['CO(GT)', 'Date', 'Time'], axis=1)
y = df_air['CO(GT)']


# Разделение на обучающую и тестовую выборки

# In[85]:


X_train, X_test, y_train, y_test = train_test_split(
X, y,
test_size=0.3,
random_state=42
)

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")


# ### Обучение модели регрессии

# Создание и обучение модели

# In[86]:


reg = DecisionTreeRegressor(
random_state=42
)

reg.fit(X_train, y_train)


# Предсказания

# In[87]:


y_pred = reg.predict(X_test)


# Оценка модели

# In[88]:


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


# In[89]:


print("Метрики регрессионного дерева:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")


# ## Часть 3: Настройка гиперпараметров для классификации
# 
# ### Использование GridSearchCV для поиска лучших параметров

# In[90]:


df = pd.read_csv('UCI_Credit_Card.csv')
df.head()


# Подготовка данных

# In[91]:


X = df.drop(['ID', 'default.payment.next.month'], axis=1)
y = df['default.payment.next.month']


# In[92]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
X, y,
test_size=0.3,
random_state=42,
stratify=y
)


# Параметры для поиска

# In[93]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

param_grid = {
'max_depth': [3, 5, 7, 10, None],
'min_samples_leaf': [1, 5, 10, 20],
'min_samples_split': [2, 10, 20]
}


# Создание и обучение GridSearchCV

# In[94]:


clf = DecisionTreeClassifier(random_state=42)
grid_search_cl = GridSearchCV(
    clf, 
    param_grid, 
    cv=5, 
    scoring='f1',
    n_jobs=-1
)
grid_search_cl.fit(X_train, y_train)

print("Лучшие параметры:")
print(grid_search_cl.best_params_)


# ### Оценка лучшей модели

# Использование лучшей модели

# In[95]:


best_clf = grid_search_cl.best_estimator_
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)


# Метрики

# In[96]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


# In[97]:


print("Метрики настройки дерева решений:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")


# ## Часть 4: Настройка гиперпараметров для регрессии
# 
# ### GridSearchCV для регрессии

# Подготовка данных

# In[98]:


df_air = pd.read_csv('AirQuality.csv', sep=';', decimal=',')


# In[99]:


df_air = df_air.dropna(axis=1, how='all')


# In[100]:


df_air.replace(-200, np.nan, inplace=True)
df_air.dropna(inplace=True)


# In[101]:


X = df_air.drop(['CO(GT)', 'Date', 'Time'], axis=1)
y = df_air['CO(GT)']


# In[102]:


X_train, X_test, y_train, y_test = train_test_split(
X, y,
test_size=0.3,
random_state=42
)


# Параметры для поиска

# In[103]:


from sklearn.tree import DecisionTreeRegressor

param_grid = {
'max_depth': [3, 5, 7, 10, None],
'min_samples_leaf': [1, 5, 10, 20]
}


# Создание и обучение GridSearchCV

# In[104]:


reg = DecisionTreeRegressor(random_state=42)
grid_search_rg = GridSearchCV(
    reg, 
    param_grid, 
    cv=5, 
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)
grid_search_rg.fit(X_train, y_train)

print("Лучшие параметры для регрессии:")
print(grid_search_rg.best_params_)


# ### Оценка лучшей регрессионной модели

# Использование лучшей модели

# In[105]:


best_reg = grid_search_rg.best_estimator_
best_reg.fit(X_train, y_train)
y_pred = best_reg.predict(X_test)


# Метрики

# In[106]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Метрики настройки регрессионного дерева:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")


# # Подготовка данных

# In[107]:


df = pd.read_csv('UCI_Credit_Card.csv')
df.head()


# In[108]:


X = df.drop(['ID', 'default.payment.next.month'], axis=1)
y = df['default.payment.next.month']


# In[109]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# ## Анализ результатов
# 
# ### Сравнение моделей классификации
