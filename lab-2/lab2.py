import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# 1. Читаем данные из датасета
df = pd.read_csv('car_sales_data.csv')

# 2. Выбираем столбец для прогнозирования (y)
y = df['Price']

# 3. Среди оставшихся столбцов найти те, чья кардинальность не превышает 8 значений
# Сначала определим категориальные столбцы с кардинальностью <= 8
categorical_cols = [cname for cname in df.columns 
                   if df[cname].nunique() <= 8 
                   and df[cname].dtype == "object"
                   and cname != 'Price']

# 4. Определяем числовые столбцы (исключая целевую переменную)
numeric_cols = [cname for cname in df.columns 
               if df[cname].dtype in ['int64', 'float64'] 
               and cname != 'Price']

# Формируем набор признаков X
my_cols = categorical_cols + numeric_cols
X = df[my_cols].copy()

# 5. Создаем объект SimpleImputer для заполнения недостающих значений в числовых столбцах
numerical_imputer = SimpleImputer(strategy='mean')

# 6. Создаем конвейер для обработки категориальных столбцов
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # заполнение модой
    ('onehot', OneHotEncoder(handle_unknown='ignore'))     # one-hot кодирование
])

# 7. Создаем препроцессор для всех выбранных столбцов
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_imputer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# 8. Создаем итоговый конвейер для RandomForest
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=0))
])

# 9. Проводим кроссвалидацию для RandomForest
print("Кросс-валидация для RandomForest:")
rf_scores = cross_val_score(rf_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
rf_mae = -rf_scores.mean()
print(f"Средняя MAE (RandomForest): {rf_mae:.2f}")
print(f"Стандартное отклонение: {rf_scores.std():.2f}")

# 10. Разделяем данные на обучающий и тестовый набор
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 11. Создаем конвейер для XGBoost
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(n_estimators=100, random_state=0))
])

# 12. Обучаем XGBoost, делаем предсказание и вычисляем MAE
print("\nОбучение XGBoost:")
xgb_pipeline.fit(X_train, y_train)
xgb_predictions = xgb_pipeline.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_predictions)
print(f"MAE (XGBoost): {xgb_mae:.2f}")

# 13. Вывод об эффективности алгоритмов
print("\nСравнение алгоритмов:")
print(f"RandomForest (кросс-валидация): {rf_mae:.2f}")
print(f"XGBoost (тестовая выборка): {xgb_mae:.2f}")

if rf_mae < xgb_mae:
    print("RandomForest показал лучшую производительность")
else:
    print("XGBoost показал лучшую производительность")