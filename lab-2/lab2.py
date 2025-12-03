import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

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

# 5. Создадим словари со стратегиями для перебора
numerical_strategies = ['mean', 'median', 'most_frequent', 'constant']
categorical_strategies = ['most_frequent', 'constant']

# 6. Функция для оценки стратегий
def evaluate_strategy(num_strategy, cat_strategy, model_type='rf'):
    """Оценивает качество модели с заданными стратегиями"""
    
    # Создаем импьютеры с текущими стратегиями
    numerical_imputer = SimpleImputer(strategy=num_strategy, fill_value=0 if num_strategy == 'constant' else None)
    categorical_imputer = SimpleImputer(strategy=cat_strategy, fill_value='missing' if cat_strategy == 'constant' else None)
    
    # Создаем конвейер для обработки категориальных столбцов
    categorical_transformer = Pipeline(steps=[
        ('imputer', categorical_imputer),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Создаем препроцессор
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_imputer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Выбираем модель для оценки
    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=50, random_state=0)  # Уменьшаем для скорости
    else:
        model = XGBRegressor(n_estimators=50, random_state=0)
    
    # Создаем пайплайн
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Оцениваем с помощью кросс-валидации
    scores = cross_val_score(pipeline, X, y, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    return -scores.mean()

# 7. Перебираем стратегии для RandomForest
print("Поиск лучших стратегий для RandomForest...")
best_score_rf = float('inf')
best_strategy_rf = {}

for num_strat in numerical_strategies:
    for cat_strat in categorical_strategies:
        try:
            score = evaluate_strategy(num_strat, cat_strat, 'rf')
            print(f"  Числовая: {num_strat:<15} Категориальная: {cat_strat:<15} MAE: {score:.2f}")
            
            if score < best_score_rf:
                best_score_rf = score
                best_strategy_rf = {'num': num_strat, 'cat': cat_strat}
        except Exception as e:
            print(f"  Числовая: {num_strat:<15} Категориальная: {cat_strat:<15} Ошибка: {str(e)[:50]}")

print(f"\nЛучшая стратегия для RandomForest:")
print(f"  Числовая: {best_strategy_rf['num']}")
print(f"  Категориальная: {best_strategy_rf['cat']}")
print(f"  Лучший MAE: {best_score_rf:.2f}\n")

# 8. Перебираем стратегии для XGBoost
print("Поиск лучших стратегий для XGBoost...")
best_score_xgb = float('inf')
best_strategy_xgb = {}

for num_strat in numerical_strategies:
    for cat_strat in categorical_strategies:
        try:
            score = evaluate_strategy(num_strat, cat_strat, 'xgb')
            print(f"  Числовая: {num_strat:<15} Категориальная: {cat_strat:<15} MAE: {score:.2f}")
            
            if score < best_score_xgb:
                best_score_xgb = score
                best_strategy_xgb = {'num': num_strat, 'cat': cat_strat}
        except Exception as e:
            print(f"  Числовая: {num_strat:<15} Категориальная: {cat_strat:<15} Ошибка: {str(e)[:50]}")

print(f"\nЛучшая стратегия для XGBoost:")
print(f"  Числовая: {best_strategy_xgb['num']}")
print(f"  Категориальная: {best_strategy_xgb['cat']}")
print(f"  Лучший MAE: {best_score_xgb:.2f}")

# 9. Создаем финальные импьютеры с лучшими стратегиями
def create_preprocessor(num_strategy, cat_strategy):
    """Создает препроцессор с заданными стратегиями"""
    # Определяем fill_value в зависимости от стратегии
    num_fill_value = 0 if num_strategy == 'constant' else None
    cat_fill_value = 'missing' if cat_strategy == 'constant' else None
    
    numerical_imputer = SimpleImputer(strategy=num_strategy, fill_value=num_fill_value)
    categorical_imputer = SimpleImputer(strategy=cat_strategy, fill_value=cat_fill_value)
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', categorical_imputer),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    return ColumnTransformer(
        transformers=[
            ('num', numerical_imputer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

# 10. Создаем препроцессоры с лучшими стратегиями для каждой модели
preprocessor_rf = create_preprocessor(best_strategy_rf['num'], best_strategy_rf['cat'])
preprocessor_xgb = create_preprocessor(best_strategy_xgb['num'], best_strategy_xgb['cat'])

# 11. Разделяем данные на обучающий и тестовый набор
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 12. Создаем и оцениваем RandomForest с лучшей стратегией
print("\n" + "="*50)
print("ОЦЕНКА RandomForest С ЛУЧШЕЙ СТРАТЕГИЕЙ")
print("="*50)

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_rf),
    ('model', RandomForestRegressor(n_estimators=100, random_state=0))
])

# Кросс-валидация для RandomForest
rf_scores = cross_val_score(rf_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
rf_mae = -rf_scores.mean()
print(f"\nКросс-валидация RandomForest:")
print(f"  Средняя MAE: {rf_mae:.2f}")
print(f"  Стандартное отклонение: {rf_scores.std():.2f}")
print(f"  Диапазон: [{(-rf_scores).min():.2f}, {(-rf_scores).max():.2f}]")

# Обучение на полном тренировочном наборе
rf_pipeline.fit(X_train, y_train)
rf_predictions = rf_pipeline.predict(X_test)
rf_test_mae = mean_absolute_error(y_test, rf_predictions)
print(f"\nТестирование на отдельной выборке:")
print(f"  MAE на тесте: {rf_test_mae:.2f}")

# 13. Создаем и оцениваем XGBoost с лучшей стратегией
print("\n" + "="*50)
print("ОЦЕНКА XGBoost С ЛУЧШЕЙ СТРАТЕГИЕЙ")
print("="*50)

xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_xgb),
    ('model', XGBRegressor(n_estimators=100, random_state=0))
])

# Кросс-валидация для XGBoost
xgb_scores = cross_val_score(xgb_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
xgb_mae = -xgb_scores.mean()
print(f"\nКросс-валидация XGBoost:")
print(f"  Средняя MAE: {xgb_mae:.2f}")
print(f"  Стандартное отклонение: {xgb_scores.std():.2f}")
print(f"  Диапазон: [{(-xgb_scores).min():.2f}, {(-xgb_scores).max():.2f}]")

# Обучение на тренировочном наборе и тестирование
xgb_pipeline.fit(X_train, y_train)
xgb_predictions = xgb_pipeline.predict(X_test)
xgb_test_mae = mean_absolute_error(y_test, xgb_predictions)
print(f"\nТестирование на отдельной выборке:")
print(f"  MAE на тесте: {xgb_test_mae:.2f}")

# 14. Сравнение алгоритмов
print("\n" + "="*50)
print("ИТОГОВОЕ СРАВНЕНИЕ")
print("="*50)

print(f"\nЛучшие стратегии заполнения пропусков:")
print(f"  RandomForest: числовая='{best_strategy_rf['num']}', категориальная='{best_strategy_rf['cat']}'")
print(f"  XGBoost: числовая='{best_strategy_xgb['num']}', категориальная='{best_strategy_xgb['cat']}'")

print(f"\nРезультаты кросс-валидации (MAE, меньше = лучше):")
print(f"  RandomForest: {rf_mae:.2f}")
print(f"  XGBoost: {xgb_mae:.2f}")

print(f"\nРезультаты на тестовой выборке:")
print(f"  RandomForest: {rf_test_mae:.2f}")
print(f"  XGBoost: {xgb_test_mae:.2f}")

if rf_mae < xgb_mae:
    print("\nВЫВОД: RandomForest показал лучшую производительность по кросс-валидации")
else:
    print("\nВЫВОД: XGBoost показал лучшую производительность по кросс-валидации")

print("\nРазница в стратегиях объясняется тем, что разные модели")
print("по-разному реагируют на способы заполнения пропущенных значений.")