#!/usr/bin/env python
# coding: utf-8

# # Лабораторная работа №3: Деревья решений и настройка гиперпараметров
# 
# ## Часть 1: Классификация с помощью дерева решений
# 
# ### Импорт необходимых библиотек

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


# ### Загрузка и предобработка данных

# Загрузка датасета по умолчанию кредитных карт
# 

# In[2]:


df = pd.read_csv('UCI_Credit_Card.csv')
df.head()


# Проверка на пропущенные значения
# 

# In[3]:


df.info()
df.isnull().sum()


# ### Подготовка данных для моделирования

# Разделение на признаки и целевую переменную

# In[4]:


X = df.drop('default.payment.next.month', axis=1)
y = df['default.payment.next.month']


# Разделение на обучающую и тестовую выборки

# In[5]:


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

# In[6]:


clf = DecisionTreeClassifier(
random_state=42
)

clf.fit(X_train, y_train)


# Предсказания

# In[7]:


y_pred = clf.predict(X_test)


# # Оценка модели

# In[8]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


# In[9]:


print("Метрики базового дерева решений:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")


# Матрица ошибок

# In[10]:


cm = confusion_matrix(y_test, y_pred)
print("Матрица ошибок:")
print(cm)


# ## Часть 2: Регрессия с помощью дерева решений
# 
# ### Загрузка и предобработка данных о качестве воздуха

# In[11]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Загрузка датасета

# In[12]:


df_air = pd.read_csv('AirQuality.csv', sep=';', decimal=',')


# Удаление полностью пустых столбцов

# In[13]:


df_air = df_air.dropna(axis=1, how='all')


# Замена значения -200 (пропуски) на NaN и удаление строк с пропусками

# In[14]:


df_air.replace(-200, np.nan, inplace=True)
df_air.dropna(inplace=True)


# ### Подготовка данных для регрессии

# Разделение на признаки и целевую переменную

# In[15]:


X = df_air.drop(['CO(GT)', 'Date', 'Time'], axis=1)
y = df_air['CO(GT)']


# Разделение на обучающую и тестовую выборки

# In[16]:


X_train, X_test, y_train, y_test = train_test_split(
X, y,
test_size=0.3,
random_state=42
)

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")


# ### Обучение модели регрессии

# Создание и обучение модели

# In[17]:


reg = DecisionTreeRegressor(
random_state=42
)

reg.fit(X_train, y_train)


# Предсказания

# In[18]:


y_pred = reg.predict(X_test)


# Оценка модели

# In[19]:


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


# In[20]:


print("Метрики регрессионного дерева:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")


# ## Часть 3: Настройка гиперпараметров для классификации
# 
# ### Использование GridSearchCV для поиска лучших параметров

# In[21]:


df = pd.read_csv('UCI_Credit_Card.csv')
df.head()


# Подготовка данных

# In[22]:


X = df.drop(['ID', 'default.payment.next.month'], axis=1)
y = df['default.payment.next.month']


# In[23]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
X, y,
test_size=0.3,
random_state=42,
stratify=y
)


# Параметры для поиска

# In[24]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

param_grid = {
'max_depth': [3, 5, 7, 10, None],
'min_samples_leaf': [1, 5, 10, 20],
'min_samples_split': [2, 10, 20]
}


# Создание и обучение GridSearchCV

# In[25]:


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

# In[26]:


best_clf = grid_search_cl.best_estimator_
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)


# Метрики

# In[27]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


# In[28]:


print("Метрики настройки дерева решений:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")


# ## Часть 4: Настройка гиперпараметров для регрессии
# 
# ### GridSearchCV для регрессии

# Подготовка данных

# In[29]:


df_air = pd.read_csv('AirQuality.csv', sep=';', decimal=',')


# In[30]:


df_air = df_air.dropna(axis=1, how='all')


# In[31]:


df_air.replace(-200, np.nan, inplace=True)
df_air.dropna(inplace=True)


# In[32]:


X = df_air.drop(['CO(GT)', 'Date', 'Time'], axis=1)
y = df_air['CO(GT)']


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(
X, y,
test_size=0.3,
random_state=42
)


# Параметры для поиска

# In[34]:


from sklearn.tree import DecisionTreeRegressor

param_grid = {
'max_depth': [3, 5, 7, 10, None],
'min_samples_leaf': [1, 5, 10, 20]
}


# Создание и обучение GridSearchCV

# In[35]:


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

# In[36]:


best_reg = grid_search_rg.best_estimator_
best_reg.fit(X_train, y_train)
y_pred = best_reg.predict(X_test)


# Метрики

# In[37]:


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

# In[38]:


df = pd.read_csv('UCI_Credit_Card.csv')
df.head()


# In[39]:


X = df.drop(['ID', 'default.payment.next.month'], axis=1)
y = df['default.payment.next.month']


# In[40]:


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

# ### Сравнение базовой и оптимизированной моделей классификации

# Обучение базового дерева решений (без подбора гиперпараметров)

# In[41]:


base_clf = DecisionTreeClassifier(random_state=42)
base_clf.fit(X_train, y_train)
y_pred_base = base_clf.predict(X_test)


# Метрики базовой модели

# In[42]:


base_accuracy = accuracy_score(y_test, y_pred_base)
base_precision = precision_score(y_test, y_pred_base)
base_recall = recall_score(y_test, y_pred_base)
base_f1 = f1_score(y_test, y_pred_base)


# Метрики оптимизированной модели (из GridSearchCV)

# In[43]:


y_pred_best = best_clf.predict(X_test)

best_accuracy = accuracy_score(y_test, y_pred_best)
best_precision = precision_score(y_test, y_pred_best)
best_recall = recall_score(y_test, y_pred_best)
best_f1 = f1_score(y_test, y_pred_best)


# Сводная таблица сравнения

# In[44]:


comparison_df = pd.DataFrame({
    'Модель': ['Базовое дерево', 'Оптимизированное дерево'],
    'Accuracy': [base_accuracy, best_accuracy],
    'Precision': [base_precision, best_precision],
    'Recall': [base_recall, best_recall],
    'F1-score': [base_f1, best_f1]
})

comparison_df


# # Выводы по сравнению моделей классификации

# По результатам эксперимента проведено сравнение базового дерева решений и модели с подобранными гиперпараметрами. Результаты представлены в таблице выше.
# Accuracy оптимизированной модели существенно выше (0.8176 против 0.7232), что указывает на общее улучшение качества классификации.
# Precision увеличилась почти в два раза (с 0.3813 до 0.6623), что означает значительное снижение числа ложноположительных предсказаний дефолта.
# Recall при этом снизилась (с 0.4033 до 0.3576), то есть оптимизированная модель стала реже находить все объекты положительного класса.
# F1-score вырос с 0.3920 до 0.4644, что говорит об улучшении баланса между точностью и полнотой.
# Таким образом, настройка гиперпараметров позволила получить более устойчивую и точную модель, ориентированную на уменьшение ложных срабатываний. Снижение recall является компромиссом, обусловленным ростом precision, и может быть допустимо в кредитном скоринге, где ложноположительные решения (выдача кредита ненадёжному заёмщику) более критичны, чем ложные отрицания.
# В целом, оптимизированное дерево решений демонстрирует лучшее обобщающее качество и является предпочтительным для практического применения.

# ### Сравнение базовой и оптимизированной моделей регрессии

# In[45]:


df_air = pd.read_csv('AirQuality.csv', sep=';', decimal=',')

df_air = df_air.dropna(axis=1, how='all')
df_air.replace(-200, np.nan, inplace=True)
df_air.dropna(inplace=True)

X_reg = df_air.drop(['CO(GT)', 'Date', 'Time'], axis=1)
y_reg = df_air['CO(GT)']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg,
    test_size=0.3,
    random_state=42
)


# Обучение базового дерева регрессии (без подбора гиперпараметров)

# In[46]:


base_reg = DecisionTreeRegressor(random_state=42)
base_reg.fit(X_train_reg, y_train_reg)
y_pred_base = base_reg.predict(X_test_reg)


# Метрики базовой модели

# In[47]:


base_mae = mean_absolute_error(y_test_reg, y_pred_base)
base_mse = mean_squared_error(y_test_reg, y_pred_base)
base_rmse = np.sqrt(base_mse)
base_r2 = r2_score(y_test_reg, y_pred_base)


# Оптимизированное дерево регрессии

# In[48]:


best_reg = grid_search_rg.best_estimator_
best_reg.fit(X_train_reg, y_train_reg)

y_pred_best = best_reg.predict(X_test_reg)


# Метрики оптимизированной модели

# In[49]:


best_mae = mean_absolute_error(y_test_reg, y_pred_best)
best_mse = mean_squared_error(y_test_reg, y_pred_best)
best_rmse = np.sqrt(best_mse)
best_r2 = r2_score(y_test_reg, y_pred_best)


# In[50]:


comparison_reg_df = pd.DataFrame({
    'Модель': ['Базовое дерево', 'Оптимизированное дерево'],
    'MAE': [base_mae, best_mae],
    'MSE': [base_mse, best_mse],
    'RMSE': [base_rmse, best_rmse],
    'R²': [base_r2, best_r2]
})

comparison_reg_df


# # Выводы по сравнению моделей регрессии

# Оптимизированная модель дерева решений показывает лучшие результаты по всем метрикам по сравнению с базовой моделью.
# 
# Средняя абсолютная ошибка (MAE) снизилась с 0.2462 до 0.2282, а значения MSE и RMSE также уменьшились, что указывает на повышение точности прогнозирования и снижение влияния крупных ошибок.
# 
# Коэффициент детерминации R² увеличился с 0.9321 до 0.9451, что говорит о лучшей способности модели объяснять вариацию целевой переменной.
# 
# Таким образом, настройка гиперпараметров позволила снизить переобучение и улучшить обобщающую способность модели регрессии, поэтому оптимизированное дерево решений является предпочтительным.
