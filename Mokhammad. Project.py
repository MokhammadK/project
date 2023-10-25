#!/usr/bin/env python
# coding: utf-8

# In[904]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.plotting import scatter_matrix
from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import ThresholdAD
from adtk.detector import OutlierDetector

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.neighbors import LocalOutlierFactor

from pmdarima import auto_arima

import warnings
warnings.filterwarnings("ignore")

from scipy.stats import zscore


# In[905]:


def sesonal(data, s):
    """
    Функція для візуалізації сезонності в часовому ряді.

    Параметри:
    data (DataFrame або Series): Часовий ряд з датами та значеннями.
    s (int): Період сезонності у часовому ряді.

    Приклад використання:
    sesonal(df, 12)  # Візуалізувати сезонність з періодом 12 (річна сезонність).
    """
    plt.figure(figsize=(19, 8), dpi=80)
    for i, y in enumerate(data.index.year.unique()):
        plt.plot(list(range(1, len(data[data.index.year == y]) + 1)), data[data.index.year == y][data.columns[0]].values, label=y)
    plt.title("Сезонність по періодам")
    plt.legend(loc="best")
    plt.show()

def metrics(real, forecast):
    """
    Функція для обчислення різних метрик для оцінки якості прогнозу часового ряду.

    Параметри:
    real (DataFrame, Series або array): Фактичні значення часового ряду.
    forecast (DataFrame, Series або array): Прогнозовані значення часового ряду.

    Приклад використання:
    metrics(real_values, forecast_values)  # Обчислити метрики для оцінки прогнозу.
    """
    
    if type(real) == pd.core.frame.DataFrame:
        real = real[real.columns[0]].values
    
    print("Тест на стаціонарність:")
    dftest = adfuller(real - forecast, autolag='AIC')
    print("\tT-статистика = {:.3f}".format(dftest[0]))
    print("\tP-значення = {:.3f}".format(dftest[1]))
    print("Критичне значення :")
    for k, v in dftest[4].items():
        print("\t{}: {} - Дані {} стаціонарні з ймовірністю {}% відсотків".format(k, v, "не" if v < dftest[0] else "", 100-int(k[:-1])))
    
    forecast = np.array(forecast)
    print('MAD:', round(abs(real - forecast).mean(), 4))
    print('MSE:', round(((real - forecast) ** 2).mean(), 4))
    print('MAPE:', round((abs(real - forecast) / real).mean(), 4))
    print('MPE:', round(((real - forecast) / real).mean(), 4))
    print('Стандартна похибка:', round(((real - forecast) ** 2).mean() ** 0.5, 4))


# In[906]:


from scipy.stats import zscore

# Функція для завантаження та обробки даних з аркуша
def process_data(file_path, sheet_name):
    # Зчитування даних з аркуша sheet_name файлу Excel
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    # Конвертуємо стовпець 'Month' у формат дати та часу
    data['Month'] = pd.to_datetime(data['Month'])
    
    # Пошук пропущених значень у стовпці 'Price'
    missing_data = data[data['Price'].isnull()]
    
    # Пошук дублікатів в усіх стовпцях
    duplicates = data[data.duplicated()]
    
    # Повертаємо оброблені дані, пропущені значення, дублікати та викиди
    return data, missing_data, duplicates

# Шлях до файлу Excel та список аркушів для обробки
file_path = 'C:\\Users\\Bruger\\Downloads\\Property.xlsx'
sheets = ['Econom', 'Comfort', 'Business']

# Основний цикл для обробки даних з різних аркушів
for sheet_name in sheets:
    # Завантаження та обробка даних для поточного аркуша
    data, missing_data, duplicates = process_data(file_path, sheet_name)
    
    # Виведення загальної інформації про поточний аркуш
    print(f"Аркуш: {sheet_name}")
    print("Загальні характеристики:")
    print(data.info())
    print("Статистичні показники:")
    print(data.describe())
    
    # Виведення пропущених значень та дублікатів
    print("Пропущені значення:")
    print(missing_data)
    print("Дублікати:")
    print(duplicates)
    
    # Обчислення з-оцінок для стовпця 'Price'
    z_scores = zscore(data['Price'])
    
    # Визначення порогу для викидів
    threshold = 3
    
    # Знаходження викидів
    outliers = data[(z_scores > threshold) | (z_scores < -threshold)]
    
    # Виведення викидів
    print("Знайдені викиди:")
    print(outliers)
    
    # Побудова графіку для даних на поточному аркуші
    plt.figure(figsize=(15, 4))
    plt.plot(data['Month'], data['Price'])
    plt.title(f'Ціни на нерухомість на первинку в Києві з 2014 по 2023 ({sheet_name})')
    plt.xlabel('Month')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()


# In[907]:


data_econom = pd.read_excel('C:\\Users\\Bruger\\Downloads\\Property.xlsx', sheet_name = 'Econom')
data_comfort = pd.read_excel('C:\\Users\\Bruger\\Downloads\\Property.xlsx', sheet_name = 'Comfort')
data_business = pd.read_excel('C:\\Users\\Bruger\\Downloads\\Property.xlsx', sheet_name = 'Business')
dataframes = [data_econom, data_comfort, data_business]

for df in dataframes:
    df['Month'] = pd.to_datetime(df['Month'])


# In[908]:


data_econom['Price'].interpolate(method='linear', inplace=True)
data_comfort['Price'].interpolate(method='linear', inplace=True)
data_business['Price'].interpolate(method='linear', inplace=True)


# In[909]:


print(data_econom.isnull().sum())
print(data_comfort.isnull().sum())
print(data_business.isnull().sum())


# In[910]:


# Фільтрація та виведення даних за 2022 рік для кожного об'єкта DataFrame
print("Дані за 2022 рік для об'єкта 'data_econom':")
print(data_econom.loc[data_econom['Month'].dt.year == 2022])
print("Дані за 2022 рік для об'єкта 'data_comfort':")
print(data_comfort.loc[data_comfort['Month'].dt.year == 2022])
print("Дані за 2022 рік для об'єкта 'data_business':")
print(data_business.loc[data_business['Month'].dt.year == 2022])


# In[911]:


# Встановлення індексу для `data_econom` на основі стовпця 'Month'
data_econom.set_index('Month', inplace=True)

# Створення об'єкта ThresholdAD з вказаними порогами для виявлення аномалій
threshold_ad = ThresholdAD(high=36700, low=11300)

# Знаходження аномалій у часовому ряді за допомогою об'єкта ThresholdAD
anomalies = threshold_ad.detect(data_econom)

# Виведення графіка часового ряду з позначками аномалій
plot(data_econom, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color='red', anomaly_tag="marker")
plt.title("Пошук аномалій для категорії 'Econom'")


# In[912]:


anomalies[anomalies.Price]


# In[913]:


# Створення об'єкту для виявлення викидів з використанням методу LOF
outlier_detector = OutlierDetector(LocalOutlierFactor(contamination=0.05))

# Виявлення аномалій у часовому ряді "data_econom"
anomalies = outlier_detector.fit_detect(data_econom)

# Візуалізація даних разом із виявленими аномаліями
plot(data_econom, anomaly=anomalies, ts_linewidth=2, anomaly_color='red', anomaly_alpha=0.3, curve_group='all')


# In[914]:


def sesonal(data_econom, s):
    # Створення нового графіка розміром 12x15 та з роздільністю 110 dpi
    plt.figure(figsize=(12, 15), dpi=110)
    
    # Ітерація через унікальні роки в індексі "data_econom"
    for i, y in enumerate(data_econom.index.year.unique()):  
        # Побудова графіку для кожного року
        # Горизонтальна вісь відображає періоди, вертикальна - значення цін
        plt.plot(list(range(1,len(data_econom[data_econom.index.year==y])+1)), data_econom[data_econom.index.year==y][data_econom.columns[0]].values, label=y)
    
    # Додавання заголовку до графіку
    plt.title("Сезонність по періодам 'Econom'")
    
    # Додавання легенди до графіку для ідентифікації років
    plt.legend(loc="best")
    
    # Відображення графіку
    plt.show()

# Виклик функції "sesonal" для аналізу сезонності
sesonal(data_econom, 12)


# In[915]:


# Використання функції seasonal_decompose для аналізу сезонності у вхідних даних "data_econom.Price"
# Модель "multiplicative" використовується для мультиплікативного розкладу, а "extrapolate_trend=12" дозволяє розширити тренд на 12 періодів
result_M = seasonal_decompose(data_econom.Price, model='multiplicative', extrapolate_trend=12)

# Оновлення параметрів графіку для встановлення розміру фігури 10x10
plt.rcParams.update({'figure.figsize': (10, 10)})

# Побудова графіків результатів аналізу сезонності та встановлення заголовку
result_M.plot().suptitle('Мультиплікативна модель для Econom')


# In[916]:


# Використання функції seasonal_decompose для аналізу сезонності у вхідних даних "data_econom.Price"
# Модель "additive" використовується для адитивного розкладу, а "extrapolate_trend=12" дозволяє розширити тренд на 12 періодів
result_M = seasonal_decompose(data_econom.Price, model='additive', extrapolate_trend=12)

# Оновлення параметрів графіку для встановлення розміру фігури 10x10
plt.rcParams.update({'figure.figsize': (10, 10)})

# Побудова графіків результатів аналізу сезонності та встановлення заголовку
result_M.plot().suptitle('Адитивна модель для Econom')


# In[917]:


# Продовжуємо виявлення аномалій та декомпозицію даних для аркушів Comfort, Business


# In[918]:


data_comfort.set_index('Month', inplace=True)
threshold_ad = ThresholdAD(high=40500, low=12100)
anomalies = threshold_ad.detect(data_comfort)
plot(data_comfort, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color='red', anomaly_tag="marker")
plt.title("Пошук аномалій для категорії 'Comfort'")

data_business.set_index('Month', inplace=True)
threshold_ad = ThresholdAD(high=75600, low=16900)
anomalies = threshold_ad.detect(data_business)
plot(data_comfort, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color='red', anomaly_tag="marker")
plt.title("Пошук аномалій для категорії 'Business'")


# In[919]:


def sesonal(data_comfort, s):
    plt.figure(figsize=(20, 30), dpi=110)
    for i, y in enumerate(data_econom.index.year.unique()):  
        plt.plot(list(range(1,len(data_comfort[data_econom.index.year==y])+1)), data_comfort[data_comfort.index.year==y][data_comfort.columns[0]].values, label=y)
    plt.title("Сезонність по періодам 'Comfort'")
    plt.legend(loc="best")
    plt.show()
sesonal(data_comfort, 12)

def sesonal(data_business, s):
    plt.figure(figsize=(20, 30), dpi=110)
    for i, y in enumerate(data_business.index.year.unique()):  
        plt.plot(list(range(1,len(data_comfort[data_business.index.year==y])+1)), data_business[data_business.index.year==y][data_business.columns[0]].values, label=y)
    plt.title("Сезонність по періодам 'Business'")
    plt.legend(loc="best")
    plt.show()
sesonal(data_business, 12)


# In[920]:


result_M = seasonal_decompose(data_comfort.Price, model='multiplicative', extrapolate_trend=12)
plt.rcParams.update({'figure.figsize': (10,10)})
result_M.plot().suptitle('Мультиплікатива модель для Comfort')

result_M = seasonal_decompose(data_business.Price, model='multiplicative', extrapolate_trend=12)
plt.rcParams.update({'figure.figsize': (10,10)})
result_M.plot().suptitle('Мультиплікатива модель для Business')


# In[921]:


result_M = seasonal_decompose(data_comfort.Price, model='additive', extrapolate_trend=12)
plt.rcParams.update({'figure.figsize': (10,10)})
result_M.plot().suptitle('Адитивна модель для Comfort')


result_M = seasonal_decompose(data_business.Price, model='additive', extrapolate_trend=12)
plt.rcParams.update({'figure.figsize': (10,10)})
result_M.plot().suptitle('Адитивна модель для Business')


# In[922]:


# Робимо прогноз для нерухомості Econom
train_econom=data_econom['2014':'2022']
train_econom.head()


# In[923]:


# Вибірка тестових даних для аналізу за 2023 рік
test_econom = data_econom['2023']
test_econom.head()


# In[924]:


# Створення та підгонка моделі SARIMA з визначеними параметрами
mod = sm.tsa.statespace.SARIMAX(train_econom, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
results = mod.fit()

# Виведення статистичної інформації про підгонку моделі
print(results.summary().tables[1])


# In[925]:


results.plot_diagnostics(figsize=(15, 12))
plt.show()


# In[926]:


# Отримання прогнозу за замовчуванням для усіх доступних даних та виведення перших 10 значень прогнозу
predict = results.get_prediction()
predicted_values = predict.predicted_mean
predicted_values[:10]


# In[927]:


# Отримання прогнозу для навчальної вибірки з початку 2015 року
predict = results.get_prediction(start='2014-10-01')
metrics(train_econom['2014-10-01':], predict.predicted_mean)


# In[928]:


# Отримання прогнозу з січня 2023 року по грудень 2025 року
predict = results.get_prediction(start='2023', end='2026')


# In[929]:


ax = data_econom.plot(figsize=(10,6), color='black', title="Прогноз методом SARIMA" )
results.fittedvalues.plot(ax=ax, style='--', color='red')
predict.predicted_mean.plot(ax=ax, style='--', color='green')
plt.show()


# In[930]:


# Отримання прогнозу з січня 2023 року по жовтень 2023 року
predict = results.get_prediction(start='2023-01-01', end='2023-10-01')

# Розрахунок метрик якості прогнозу для порівняння зі змінними test_econom
metrics(test_econom, predict.predicted_mean)


# In[931]:


# Отримання прогнозу
predict = results.get_prediction(start='2023', end='2026')

# Отримання середніх значень прогнозу
predicted_mean = predict.predicted_mean

# Виведення середніх значень прогнозу
print(predicted_mean)


# In[953]:


"""У 2023 році спостерігається зростання середньомісячної ціни, починаючи з 32,714.92 і закінчуючи 38,567.53 в грудні 2023 року.
В 2024 році прогнозується подальше зростання цін, протягом цього року середньомісячна ціна зросте з 39,710.04 в січні до 50,819.44 в жовтні.
У 2025 році також передбачається зростання цін, і середньомісячна ціна збільшиться з 51,635.57 в січні до 59,669.41 в грудні.
В 2026 році прогнозується подальше зростання цін, протягом року середньомісячна ціна зросте з 60,688.82 в січні до нового рівня.
"""


# In[932]:


# Робимо прогноз для нерухомості Comfort
train_comfort=data_comfort['2014':'2022']
train_comfort.head()


# In[933]:


test_comfort = data_comfort['2023']
test_comfort.head()


# In[934]:


mod = sm.tsa.statespace.SARIMAX(train_comfort, order=(1, 0, 0), seasonal_order=(1, 1, 0, 12))
results = mod.fit()
print(results.summary().tables[1])


# In[935]:


results.plot_diagnostics(figsize=(15, 12))
plt.show()


# In[936]:


predict = results.get_prediction()
predicted_values = predict.predicted_mean
predicted_values[:10]


# In[937]:


predict = results.get_prediction(start='2014-10-01')
metrics(train_econom['2014-10-01':], predict.predicted_mean)


# In[938]:


predict = results.get_prediction(start='2023', end='2026')


# In[939]:


ax = data_comfort.plot(figsize=(10,6), color='black', title="Прогноз методом SARIMA" )
results.fittedvalues.plot(ax=ax, style='--', color='red')
predict.predicted_mean.plot(ax=ax, style='--', color='green')
plt.show()


# In[940]:


predict = results.get_prediction(start='2023-01-01', end='2023-10-01')
metrics(test_comfort, predict.predicted_mean)


# In[941]:


predict = results.get_prediction(start='2023', end='2026')
predicted_mean = predict.predicted_mean
print(predicted_mean)


# In[954]:


"""
З січня 2023 року по грудень 2023 року спостерігалося значне зменшення середньомісячної ціни, зі значенням 36,627.97 в січні до 36,563.66 в грудні 2023 року.
У 2024 році ціни почали зростати, зі значенням 35,734.50 в січні, до 38,071.54 в грудні 2024 року.
У 2025 році також спостерігалося збільшення середньомісячної ціни, з 37,036.05 в січні до 37,730.25 в березні, та подальше збільшення до 36,730.25 в січні 2026 року.
"""


# In[942]:


# Робимо прогноз для нерухомості Business
train_business=data_business['2014':'2022']
train_business.head()


# In[943]:


test_business = data_business['2023']
test_business.head()


# In[944]:


mod = sm.tsa.statespace.SARIMAX(train_business, order=(1, 0, 1), seasonal_order=(1, 1, 0, 12))
results = mod.fit()
print(results.summary().tables[1])


# In[945]:


results.plot_diagnostics(figsize=(15, 12))
plt.show()


# In[946]:


predict = results.get_prediction()
predicted_values = predict.predicted_mean
predicted_values[:10]


# In[947]:


predict = results.get_prediction(start='2014-10-01')
metrics(train_business['2014-10-01':], predict.predicted_mean)


# In[948]:


predict = results.get_prediction(start='2023', end='2026')


# In[949]:


ax = data_business.plot(figsize=(10,6), color='black', title="Прогноз методом SARIMA" )
results.fittedvalues.plot(ax=ax, style='--', color='red')
predict.predicted_mean.plot(ax=ax, style='--', color='green')
plt.show()


# In[950]:


predict = results.get_prediction(start='2023-01-01', end='2023-10-01')
metrics(test_business, predict.predicted_mean)


# In[951]:


predict = results.get_prediction(start='2023', end='2026')
predicted_mean = predict.predicted_mean
print(predicted_mean)


# In[955]:


"""
З січня 2023 року по грудень 2023 року спостерігалося збільшення середньомісячної ціни, зі значенням 69,469.47 в січні до 72,826.41 в грудні 2023 року.
У 2024 році спостерігається подальше збільшення цін, починаючи з 71,535.51 в січні і досягаючи 76,938.47 в грудні 2024 року.
У 2025 році знову спостерігається збільшення цін, з 75,360.33 в січні до 77,297.65 в грудні 2025 року.
"""


# In[ ]:




