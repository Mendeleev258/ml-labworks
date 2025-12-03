import pandas as pd
import numpy as np

# Читаем исходный файл
df = pd.read_csv('car_sales_data.csv')

# Добавляем пропуски в числовые столбцы (кроме Price)
numeric_cols = [cname for cname in df.columns 
               if df[cname].dtype in ['int64', 'float64'] 
               and cname != 'Price']

for col in numeric_cols:
    # 10% пропусков в числовых данных
    mask = np.random.rand(len(df)) < 0.10
    df.loc[mask, col] = np.nan

# Добавляем пропуски в категориальные столбцы (до 8 уникальных значений)
categorical_cols = [cname for cname in df.columns 
                   if df[cname].nunique() <= 8 
                   and df[cname].dtype == "object"
                   and cname != 'Price']

for col in categorical_cols:
    # 5% пропусков в категориальных данных
    mask = np.random.rand(len(df)) < 0.05
    df.loc[mask, col] = np.nan

# Сохраняем в новый файл
df.to_csv('car_sales_data_with_missing.csv', index=False)

print("Файл с пропусками сохранен как 'car_sales_data_with_missing.csv'")
print(f"Всего пропусков: {df.isnull().sum().sum()}")