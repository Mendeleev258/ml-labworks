import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# путь к датасету
data_file_path = 'car_sales_data.csv'

# прочитать данные в переменную типа DataFrame
cars_data = pd.read_csv(data_file_path)

# размер датасета
print("Размер датасета:", cars_data.shape)

# печать всех данных
print("\nОписание данных:")
print(cars_data.describe())

print("\nПервые 5 строк:")
print(cars_data.head())

# Кодируем категориальные переменные
label_encoders = {}
categorical_columns = ['Manufacturer', 'Model', 'Fuel type']

for column in categorical_columns:
    le = LabelEncoder()
    cars_data[column + '_encoded'] = le.fit_transform(cars_data[column])
    label_encoders[column] = le

# выбираем данные для прогнозирования
y = cars_data.Price

# список столбцов для выборки (используем закодированные версии)
features = ['Manufacturer_encoded', 'Model_encoded', 'Engine size', 'Fuel type_encoded', 
           'Year of manufacture', 'Mileage']

# сама выборка
X = cars_data[features]

print("\nПризнаки после кодирования:")
print(X.head())
print("\nЦелевая переменная (Price):")
print(y.head())

# РАЗДЕЛЯЕМ ДАННЫЕ ДО ОБУЧЕНИЯ
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=1)

print(f"\nРазмер тренировочной выборки: {train_X.shape}")
print(f"Размер тестовой выборки: {test_X.shape}")

# Определяем и обучаем модель на тренировочных данных
cars_model = DecisionTreeRegressor(random_state=1)
cars_model.fit(train_X, train_y)

print("\n" + "="*50)
print("ПРОГНОЗ НА ТЕСТОВЫХ ДАННЫХ (которые модель не видела):")
print("="*50)

print("\nПервые 5 записей тестовой выборки:")
print(test_X.head())
print("\nПрогноз цен:")
test_predictions = cars_model.predict(test_X.head())
print(test_predictions)
print("\nРеальные цены:")
print(test_y.head().values)

# Считаем ошибку на тестовых данных
test_mae = mean_absolute_error(test_y, cars_model.predict(test_X))
print(f"\nMAE на тестовых данных: {test_mae:.2f}")

# Для сравнения - MAE на тренировочных данных
train_predictions = cars_model.predict(train_X)
train_mae = mean_absolute_error(train_y, train_predictions)
print(f"MAE на тренировочных данных: {train_mae:.2f}")

print("\n" + "="*50)
print("ОПТИМИЗАЦИЯ ПАРАМЕТРОВ ДЕРЕВА РЕШЕНИЙ")
print("="*50)

# Функция для подсчета средней абсолютной ошибки
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

print("\nСравнение MAE для разных глубин дерева (на валидационной выборке):")
best_mae = float('inf')
best_depth = 0

for max_leaf_nodes in [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]:
    my_mae = get_mae(max_leaf_nodes, train_X, test_X, train_y, test_y)
    print("Макс. листьев: %d  \t\t MAE:  %.2f" % (max_leaf_nodes, my_mae))
    
    if my_mae < best_mae:
        best_mae = my_mae
        best_depth = max_leaf_nodes

print(f"\nЛучший параметр: max_leaf_nodes = {best_depth} с MAE = {best_mae:.2f}")

print("\n" + "="*50)
print("МОДЕЛЬ СЛУЧАЙНОГО ЛЕСА")
print("="*50)

# Случайный лес
forest_model = RandomForestRegressor(n_estimators=100, random_state=1)
forest_model.fit(train_X, train_y)

forest_train_predictions = forest_model.predict(train_X)
forest_train_mae = mean_absolute_error(train_y, forest_train_predictions)
print(f"MAE RandomForest на тренировочных данных: {forest_train_mae:.2f}")

forest_test_predictions = forest_model.predict(test_X)
forest_test_mae = mean_absolute_error(test_y, forest_test_predictions)
print(f"MAE RandomForest на тестовых данных: {forest_test_mae:.2f}")

print("\n" + "="*50)
print("СРАВНЕНИЕ МОДЕЛЕЙ")
print("="*50)

print(f"Decision Tree тест MAE: {test_mae:.2f}")
print(f"Random Forest тест MAE: {forest_test_mae:.2f}")