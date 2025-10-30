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

# Определяем модель
cars_model = DecisionTreeRegressor(random_state=1)

# Обучаем модель
cars_model.fit(X, y)

print("\nПрогноз будет составлен для 5 первых записей:")
print(X.head())
print("Прогноз:")
print(cars_model.predict(X.head()))
print("Реальные данные:")
print(y.head().values)

from sklearn.metrics import mean_absolute_error

predicted_car_prices = cars_model.predict(X)
print("\nMAE на всех данных: %d" % mean_absolute_error(y, predicted_car_prices))

# разделяем данные на тренировочный и тестовый набор
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# определяем модель
cars_model = DecisionTreeRegressor(max_leaf_nodes=10, random_state=0)

# обучаем модель
cars_model.fit(train_X, train_y)

# получаем среднюю абсолютную ошибку
val_predictions = cars_model.predict(val_X)
print("MAE на тестовых данных: %d" % mean_absolute_error(val_y, val_predictions))

# функция для подсчета средней абсолютной ошибки на заданных данных для заданной глубины дерева
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

# сравнение MAE для разной глубины дерева
print("\nСравнение MAE для разных глубин дерева:")
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Макс. листьев: %d  \t\t MAE:  %d" % (max_leaf_nodes, my_mae))

# случайный лес
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
forest_preds = forest_model.predict(val_X)
print("\nMAE RandomForest: %d" % mean_absolute_error(val_y, forest_preds))