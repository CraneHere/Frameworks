#!/usr/bin/env python
# coding: utf-8

# # 1. Выбор начальных условий

# ## 1.1 Задача классификации

# Датасет:
# Default of Credit Card Clients Dataset
# Постановка задачи:
# Бинарная классификация — определить, будет ли клиент иметь дефолт по кредитной карте в следующем месяце.
# Почему это реальная задача:
# Подобные модели используются банками для:
# оценки кредитных рисков
# автоматического принятия решений по выдаче кредитов
# снижения финансовых потерь
# Целевая переменная:
# default.payment.next.month (0 — нет дефолта, 1 — дефолт)

# ## 1.2 Задача регрессии

#  Датасет:
# Air Quality Dataset
# Постановка задачи:
# Предсказание концентрации загрязнителя воздуха (например, CO(GT)) на основе метеоданных и показаний сенсоров.
# Почему это реальная задача:
# мониторинг качества воздуха
# прогноз загрязнений
# экология и городское планирование
# Целевая переменная:
# CO(GT) — концентрация угарного газа

# # 2. Выбор метрик качества

# ## Для классификации:

# Accuracy — общая точность
# ROC-AUC — качество разделения классов (важно при дисбалансе)
# Обоснование:
# В задаче дефолта важно не только общее количество верных ответов, но и способность модели различать классы.

# ## Для регрессии:

# MAE (Mean Absolute Error) — средняя абсолютная ошибка
# RMSE (Root Mean Squared Error) — штрафует большие ошибки
# Обоснование:
# Эти метрики хорошо интерпретируемы и широко применяются в задачах прогнозирования.

# # 3. Создание бейзлайна для классификации

# In[1]:


## Импортируем необходимые библиотеки для работы с данными и KNN.
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


# In[2]:


## Загружаем датасет и смотрим на структуру данных.
df = pd.read_csv("UCI_Credit_Card.csv")
df.head()


# In[3]:


df.info()


# In[4]:


## Разделяем признаки и целевую переменную.
X = df.drop("default.payment.next.month", axis=1)
y = df["default.payment.next.month"]


# In[5]:


## Делим данные на обучающую и тестовую выборки.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[6]:


## Обучаем KNN с базовым параметром k=5.
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train, y_train)


# In[7]:


## Оцениваем качество бейзлайн-модели по выбранным метрикам.
y_pred = knn_clf.predict(X_test)
y_proba = knn_clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

accuracy, roc_auc


# # 4. Улучшение бейзлайна
# ## Гипотезы:
# ### KNN чувствителен к масштабу → нормализация признаков улучшит качество
# ### Подбор оптимального k через кросс-валидацию повысит метрики

# In[8]:


## Используем StandardScaler для нормализации данных.
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[9]:


## Создаем pipeline: масштабирование + KNN.
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])


# In[10]:


## Определяем сетку гиперпараметров.
from sklearn.model_selection import GridSearchCV

param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 11],
    "knn__weights": ["uniform", "distance"]
}


# In[11]:


## Выполняем кросс-валидацию для поиска лучших параметров.
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="roc_auc"
)

grid.fit(X_train, y_train)


# In[12]:


## Оцениваем улучшенный бейзлайн.
best_model = grid.best_estimator_

y_pred_best = best_model.predict(X_test)
y_proba_best = best_model.predict_proba(X_test)[:, 1]

accuracy_best = accuracy_score(y_test, y_pred_best)
roc_auc_best = roc_auc_score(y_test, y_proba_best)

accuracy_best, roc_auc_best


# * Масштабирование признаков существенно улучшило качество модели, поскольку KNN чувствителен к разным масштабам данных
# * Подбор гиперпараметров через GridSearchCV позволил найти более оптимальные значения k и весов
# * ROC-AUC улучшился значительно, что говорит о лучшей способности модели разделять классы (дефолт / не дефолт)

# # 5. Создание бейзлайна для регрессии

# In[13]:


df_air = pd.read_csv("AirQuality.csv", sep=";", decimal=",")
df_air.head()


# In[14]:


df_air.info()


# In[15]:


## Удаляем пустые столбцы
df_air = df_air.dropna(axis=1, how='all')

## Заменяем -200 (пропуски) на NaN
df_air = df_air.replace(-200, np.nan)

## Удаляем строки с пропусками
df_air = df_air.dropna()


# In[16]:


## Выделяем признаки и целевую переменную.
X = df_air.drop(['CO(GT)', 'Date', 'Time'], axis=1)
y = df_air['CO(GT)']


# In[17]:


## Делим данные на train/test.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[18]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[19]:


## Обучаем KNN-регрессию с базовыми параметрами.
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train, y_train)


# In[20]:


## Оцениваем качество бейзлайна.
y_pred = knn_reg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

mae, rmse


# # 6. Улучшение бейзлайна 
# ## Гипотезы:
# ### Масштабирование улучшит расстояния
# ### Подбор k снизит ошибку

# In[21]:


# Создаём pipeline, который:
# стандартизирует признаки (нулевое среднее и единичная дисперсия),
# обучает модель KNN-регрессии на масштабированных данных.
pipeline_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor())
])


# In[22]:


# Импортируем инструмент для перебора гиперпараметров с кросс-валидацией.
param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 11],
    "knn__weights": ["uniform", "distance"]
}


# In[23]:


# Задаём сетку гиперпараметров:
# количество соседей k,
# способ взвешивания соседей.
grid_reg = GridSearchCV(
    pipeline_reg,
    param_grid,
    cv=5,
    scoring="neg_mean_absolute_error"
)

grid_reg.fit(X_train, y_train)


# In[24]:


# Оцениваем качество улучшенного бейзлайна по метрикам MAE и RMSE.
best_reg = grid_reg.best_estimator_

y_pred_best = best_reg.predict(X_test)

mae_best = mean_absolute_error(y_test, y_pred_best)
rmse_best = mean_squared_error(y_test, y_pred_best, squared=False)

mae_best, rmse_best


# * Нормализация данных оказалась важной для регрессии KNN
# * Оптимизация гиперпараметров позволила снизить обе метрики ошибки (MAE и RMSE)
# * Уменьшение RMSE означает, что модель стала лучше справляться с большими отклонениями в прогнозах

# # 7. Имплементация алгоритма KNN

# ## 7.1 Реализация KNN для классификации

# In[ ]:


df_credit = pd.read_csv("UCI_Credit_Card.csv")
df_credit.head()


# In[ ]:


X_clf = df_credit.drop("default.payment.next.month", axis=1)
y_clf = df_credit["default.payment.next.month"]


# In[ ]:


X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.25, random_state=42, stratify=y_clf
)


# In[ ]:


df_air = pd.read_csv('AirQuality.csv', sep=';', decimal=',')

df_air = df_air.dropna(axis=1, how='all')

df_air = df_air.drop(columns=["Date", "Time"]).fillna(method="ffill")

df_air.replace(-200, np.nan, inplace=True)
df_air.dropna(inplace=True)

df_air.head()


# In[ ]:


target = "CO(GT)"
X_reg = df_air.drop(columns=[target])
y_reg = df_air[target]


# In[ ]:


X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.25, random_state=42
)


# In[ ]:


class KNNClassifierCustom:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X.values if hasattr(X, "values") else X
        self.y_train = y.values if hasattr(y, "values") else y

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        X = X.values if hasattr(X, "values") else X
        predictions = [self._predict_one(x) for x in X]
        return np.array(predictions)

    def predict_proba(self, X):
        X = X.values if hasattr(X, "values") else X
        probas = [self._predict_proba_one(x) for x in X]
        return np.array(probas)

    def _predict_one(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        return Counter(k_labels).most_common(1)[0][0]

    def _predict_proba_one(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        prob_1 = np.mean(k_labels)
        return [1 - prob_1, prob_1]


# ## 4.2 Реализация KNN для регрессии

# In[ ]:


class KNNRegressor:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict_one(x) for x in X]
        return np.array(predictions)

    def _predict_one(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_values = [self.y_train[i] for i in k_indices]
        return np.mean(k_values)


# ## Обучение и оценка (классификация)

# In[ ]:


rf_clf_custom = KNNClassifierCustom(n_estimators=100, max_depth=None, random_state=42)
rf_clf_custom.fit(X_clf_train.values, y_clf_train.values)
y_pred_clf_custom = rf_clf_custom.predict(X_clf_test.values)


# ## Обучение и оценка (регрессия)
