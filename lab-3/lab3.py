import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow import keras
from keras import layers, callbacks
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ============================================

print("=" * 60)
print("1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
print("=" * 60)

# Загружаем данные
df = pd.read_csv('car_sales_data.csv')
print(f"Размер датасета: {df.shape}")
print(f"Столбцы: {list(df.columns)}")
print("\nПервые 5 строк:")
print(df.head())

# 2. Подготовка данных для модели
# Целевая переменная - Price
y = df['Price']

# Выбираем признаки
# Категориальные признаки с малой кардинальностью
categorical_cols = ['Fuel type', 'Manufacturer']
# Числовые признаки
numeric_cols = ['Engine size', 'Year of manufacture', 'Mileage']

# Создаем X
X = df[categorical_cols + numeric_cols].copy()

# Проверяем пропуски
print("\nПроверка пропущенных значений:")
print(X.isnull().sum())

# Если есть пропуски - заполняем
if X.isnull().sum().sum() > 0:
    # Для числовых - медианой
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())
    # Для категориальных - модой
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

# Препроцессинг: OneHot для категориальных, StandardScaler для числовых
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Преобразуем данные
X_processed = preprocessor.fit_transform(X)

# Разделяем на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Дополнительно разделяем тренировочную на train/validation
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print(f"\nРазмеры выборок:")
print(f"X_train: {X_train_final.shape}")
print(f"X_val: {X_val.shape}")
print(f"X_test: {X_test.shape}")

# ============================================
# 3. ОПРЕДЕЛЕНИЕ 5 МОДЕЛЕЙ НЕЙРОННЫХ СЕТЕЙ
# ============================================

print("\n" + "=" * 60)
print("2. ОПИСАНИЕ 5 МОДЕЛЕЙ НЕЙРОННЫХ СЕТЕЙ")
print("=" * 60)

# Определяем размер входа (после препроцессинга)
input_shape = X_train_final.shape[1]
print(f"Размер входного вектора: {input_shape}")

# Модель 1: Простая сеть (базовая)
def create_model_1():
    """Простая нейронная сеть с 2 слоями"""
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[input_shape]),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Выходной слой для регрессии
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',  # MSE для регрессии
        metrics=['mae']  # MAE как дополнительная метрика
    )
    return model

# Модель 2: Более глубокая сеть
def create_model_2():
    """Глубокая нейронная сеть с 4 слоями"""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=[input_shape]),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

# Модель 3: Широкая сеть
def create_model_3():
    """Широкая нейронная сеть с большим количеством нейронов"""
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=[input_shape]),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae']
    )
    return model

# Модель 4: Сеть с регуляризацией Dropout
def create_model_4():
    """Сеть с Dropout для борьбы с переобучением"""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=[input_shape]),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

# Модель 5: Сеть с BatchNormalization
def create_model_5():
    """Сеть с BatchNormalization для ускорения обучения"""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=[input_shape]),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(1)
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

# Создаем список моделей
models = [
    ("Модель 1: Простая сеть (2 слоя)", create_model_1()),
    ("Модель 2: Глубокая сеть (4 слоя)", create_model_2()),
    ("Модель 3: Широкая сеть", create_model_3()),
    ("Модель 4: Сеть с Dropout", create_model_4()),
    ("Модель 5: Сеть с BatchNorm", create_model_5())
]

print("\nСоздано 5 моделей нейронных сетей:")
for name, model in models:
    print(f"\n{name}:")
    model.summary()

# ============================================
# 4. ОБУЧЕНИЕ МОДЕЛЕЙ И ПОСТРОЕНИЕ ГРАФИКОВ
# ============================================

print("\n" + "=" * 60)
print("3. ОБУЧЕНИЕ МОДЕЛЕЙ И АНАЛИЗ РЕЗУЛЬТАТОВ")
print("=" * 60)

# Создаем Early Stopping callback
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    min_delta=0.001,
    restore_best_weights=True,
    verbose=1
)

# Словарь для хранения истории обучения
histories = {}

# Обучаем каждую модель
for name, model in models:
    print(f"\nОбучение {name}...")
    
    history = model.fit(
        X_train_final, y_train_final,
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=150,
        callbacks=[early_stopping],
        verbose=0  # Не выводить прогресс каждой эпохи
    )
    
    histories[name] = history
    print(f"  Обучено эпох: {len(history.history['loss'])}")
    print(f"  Final Train Loss: {history.history['loss'][-1]:.2f}")
    print(f"  Final Val Loss: {history.history['val_loss'][-1]:.2f}")

# ============================================
# 5. ПОСТРОЕНИЕ ГРАФИКОВ ПОТЕРЬ
# ============================================

print("\n" + "=" * 60)
print("4. ГРАФИКИ ПОТЕРЬ ДЛЯ КАЖДОЙ МОДЕЛИ")
print("=" * 60)

# Создаем большую фигуру с 5 подграфиками
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# Цвета для графиков
colors = ['blue', 'red', 'green', 'orange', 'purple']

# Рисуем графики для каждой модели
for idx, (name, history) in enumerate(histories.items()):
    if idx >= len(axes):
        break
        
    ax = axes[idx]
    
    # График потерь на тренировочной выборке
    ax.plot(history.history['loss'], 
            label='Train Loss', 
            color=colors[idx], 
            linewidth=2)
    
    # График потерь на валидационной выборке
    ax.plot(history.history['val_loss'], 
            label='Val Loss', 
            color=colors[idx], 
            linestyle='--',
            linewidth=2)
    
    ax.set_title(name, fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Добавляем лучшую эпоху (минимум val_loss)
    best_epoch = np.argmin(history.history['val_loss'])
    best_val_loss = history.history['val_loss'][best_epoch]
    ax.axvline(x=best_epoch, color='red', linestyle=':', alpha=0.5)
    ax.text(best_epoch, best_val_loss, 
            f'Best: {best_val_loss:.2f}',
            fontsize=10, color='red',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Удаляем пустой подграфик (если 5 моделей в 6 ячейках)
if len(models) < 6:
    axes[-1].axis('off')

# Общий график сравнения всех моделей
fig2, ax2 = plt.subplots(figsize=(12, 6))

for idx, (name, history) in enumerate(histories.items()):
    # Берем только val_loss для сравнения
    val_loss = history.history['val_loss']
    epochs = range(1, len(val_loss) + 1)
    
    ax2.plot(epochs, val_loss, 
             label=name, 
             color=colors[idx],
             linewidth=2,
             alpha=0.8)

ax2.set_title('Сравнение валидационных потерь всех моделей', 
              fontsize=16, fontweight='bold')
ax2.set_xlabel('Эпоха', fontsize=14)
ax2.set_ylabel('Validation Loss (MSE)', fontsize=14)
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')  # Логарифмическая шкала для лучшей визуализации

plt.tight_layout()
plt.show()

# ============================================
# 6. АНАЛИЗ И ВЫВОДЫ
# ============================================

print("\n" + "=" * 60)
print("5. АНАЛИЗ ЭФФЕКТИВНОСТИ МОДЕЛЕЙ")
print("=" * 60)

# Собираем метрики для сравнения
results = []
for name, history in histories.items():
    # Наименьшая валидационная ошибка
    min_val_loss = min(history.history['val_loss'])
    min_train_loss = min(history.history['loss'])
    
    # Эпоха с лучшей валидацией
    best_epoch = np.argmin(history.history['val_loss']) + 1
    
    # Разница между train и val loss (переобучение)
    overfitting_ratio = history.history['loss'][-1] / history.history['val_loss'][-1]
    
    results.append({
        'Модель': name,
        'Лучшая Val Loss': min_val_loss,
        'Лучшая Train Loss': min_train_loss,
        'Эпоха с лучшим val_loss': best_epoch,
        'Переобучение (train/val)': overfitting_ratio
    })

# Создаем DataFrame с результатами
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Лучшая Val Loss')

print("\nСравнение моделей (от лучшей к худшей):")
print(results_df.to_string(index=False))

# Определяем лучшую модель
best_model_name = results_df.iloc[0]['Модель']
best_val_loss = results_df.iloc[0]['Лучшая Val Loss']

print(f"\n{'='*60}")
print("ВЫВОДЫ:")
print(f"{'='*60}")
print(f"1. Лучшая модель: {best_model_name}")
print(f"   с Validation Loss: {best_val_loss:.4f}")
print(f"\n2. Анализ переобучения:")
for _, row in results_df.iterrows():
    ratio = row['Переобучение (train/val)']
    if ratio < 0.8:
        status = "✓ НЕДООБУЧЕНИЕ (train loss > val loss)"
    elif ratio > 1.2:
        status = "✗ ПЕРЕОБУЧЕНИЕ (train loss < val loss)"
    else:
        status = "✓ БАЛАНС (хорошая обобщающая способность)"
    print(f"   {row['Модель'][:20]:20} {ratio:.2f} - {status}")

print(f"\n3. Рекомендации:")
print("   - Модели с регуляризацией (Dropout) обычно лучше обобщают")
print("   - Слишком глубокие сети могут переобучаться на небольших данных")
print("   - BatchNormalization ускоряет обучение и улучшает стабильность")

# ============================================
# 7. ТЕСТИРОВАНИЕ ЛУЧШЕЙ МОДЕЛИ
# ============================================

print(f"\n{'='*60}")
print("6. ТЕСТИРОВАНИЕ ЛУЧШЕЙ МОДЕЛИ НА ТЕСТОВОЙ ВЫБОРКЕ")
print(f"{'='*60}")

# Находим лучшую модель
best_model_idx = results_df.index[0]
best_model_name, best_model = models[best_model_idx]

# Предсказания на тестовой выборке
y_pred = best_model.predict(X_test)

# Рассчитываем метрики
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nМодель: {best_model_name}")
print(f"Test MSE: {mse:.2f}")
print(f"Test MAE: {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# Визуализация предсказаний vs реальные значения
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot предсказаний
ax1.scatter(y_test, y_pred, alpha=0.5, color='blue')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Идеальная линия')
ax1.set_xlabel('Реальные цены', fontsize=12)
ax1.set_ylabel('Предсказанные цены', fontsize=12)
ax1.set_title(f'Предсказания vs Реальные значения\n{best_model_name}', 
              fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Распределение ошибок
errors = y_pred.flatten() - y_test.values
ax2.hist(errors, bins=50, color='green', alpha=0.7, edgecolor='black')
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Ошибка предсказания', fontsize=12)
ax2.set_ylabel('Частота', fontsize=12)
ax2.set_title('Распределение ошибок', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("ЛАБОРАТОРНАЯ РАБОТА ВЫПОЛНЕНА!")
print("="*60)