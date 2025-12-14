# ============================================================================
# ОПТИМИЗИРОВАННАЯ ВЕРСИЯ: Эффективная загрузка данных
# ============================================================================
import os, warnings
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import layers
import pandas as pd
import time

# Настройки воспроизводимости
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed()

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore")

print("="*60)
print("ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
print("="*60)

# 1. ЗАГРУЗКА ДАННЫХ С ИСПОЛЬЗОВАНИЕМ tf.data
start_time = time.time()

train_ds, valid_ds = tfds.load(
    'rock_paper_scissors',
    split=['train', 'test'],
    as_supervised=True,
    shuffle_files=True
)

print("Датасет загружен за {:.1f} секунд".format(time.time() - start_time))

# 2. ПРЕДОБРАБОТКА ДАННЫХ
def preprocess(image, label, target_size=(128, 128)):
    image = tf.image.resize(image, target_size)
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.one_hot(label, depth=3)
    return image, label

batch_size = 32
train_dataset = (
    train_ds
    .map(lambda x, y: preprocess(x, y, target_size=(128, 128)))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

valid_dataset = (
    valid_ds
    .map(lambda x, y: preprocess(x, y, target_size=(128, 128)))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

print("Данные подготовлены за {:.1f} секунд".format(time.time() - start_time))
print("Batch size: {}".format(batch_size))
print("Размер изображений: 128x128x3")

# 3. ПРОВЕРКА ДАННЫХ
print("\nПроверка данных...")
for images, labels in train_dataset.take(1):
    print("  Размер батча: {}".format(images.shape))
    print("  Диапазон значений: [{:.3f}, {:.3f}]".format(
        tf.reduce_min(images).numpy(), 
        tf.reduce_max(images).numpy()))
    print("Данные загружены корректно")
    break

# ============================================================================
# 2. СОЗДАНИЕ 3 МОДЕЛЕЙ СВЕРТОЧНЫХ СЕТЕЙ
# ============================================================================
print("\n" + "="*60)
print("СОЗДАНИЕ 3 МОДЕЛЕЙ CNN")
print("="*60)

# МОДЕЛЬ 1: Компактная архитектура
model1 = keras.Sequential([
    layers.Conv2D(16, 3, activation='relu', padding='same', input_shape=(128, 128, 3)),
    layers.MaxPool2D(2),
    
    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.MaxPool2D(2),
    
    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.MaxPool2D(2),
    
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')
])

# МОДЕЛЬ 2: Архитектура со средним количеством параметров
model2 = keras.Sequential([
    layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(128, 128, 3)),
    layers.MaxPool2D(2),
    
    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.MaxPool2D(2),
    
    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.MaxPool2D(2),
    
    layers.Conv2D(128, 3, activation='relu', padding='same'),
    layers.GlobalAveragePooling2D(),
    
    layers.Dense(3, activation='softmax')
])

# МОДЕЛЬ 3: Архитектура с BatchNormalization
model3 = keras.Sequential([
    layers.Conv2D(32, 3, padding='same', input_shape=(128, 128, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D(2),
    
    layers.Conv2D(64, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D(2),
    
    layers.Conv2D(128, 3, padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D(2),
    
    layers.GlobalAveragePooling2D(),
    layers.Dense(3, activation='softmax')
])

models = [
    ("Модель 1: Компактная CNN", model1),
    ("Модель 2: CNN со средним количеством параметров", model2),
    ("Модель 3: CNN с BatchNormalization", model3)
]

for name, model in models:
    print("\n{}:".format(name))
    print("  Количество параметров: {:,}".format(model.count_params()))

# ============================================================================
# 3. ОБУЧЕНИЕ МОДЕЛЕЙ
# ============================================================================
print("\n" + "="*60)
print("ОБУЧЕНИЕ МОДЕЛЕЙ")
print("="*60)

histories = {}

# Callback для ранней остановки
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

for name, model in models:
    print("\n" + "-"*40)
    print("Обучение {}".format(name))
    print("-"*40)
    
    start_train = time.time()
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=20,
        verbose=1,
        callbacks=[early_stopping]
    )
    
    training_time = time.time() - start_train
    histories[name] = history.history  # Сохраняем history.history, а не сам объект
    
    final_acc = history.history['val_accuracy'][-1]
    print("Обучение завершено за {:.1f} секунд".format(training_time))
    print("Финальная точность на валидации: {:.4f}".format(final_acc))
    print("Количество эпох обучения: {}".format(len(history.history['loss'])))

# ============================================================================
# 4. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# ============================================================================
print("\n" + "="*60)
print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['blue', 'red', 'green']

# Собираем данные для графиков
for idx, (name, history) in enumerate(histories.items()):
    epochs = range(1, len(history['loss']) + 1)
    
    axes[0].plot(epochs, history['loss'], 
                color=colors[idx], linewidth=2, alpha=0.8, label='{} (train)'.format(name))
    axes[0].plot(epochs, history['val_loss'], 
                color=colors[idx], linestyle='--', linewidth=2, alpha=0.8, label='{} (val)'.format(name))
    
    axes[1].plot(epochs, history['accuracy'], 
                color=colors[idx], linewidth=2, alpha=0.8)
    axes[1].plot(epochs, history['val_accuracy'], 
                color=colors[idx], linestyle='--', linewidth=2, alpha=0.8)

axes[0].set_xlabel('Эпоха')
axes[0].set_ylabel('Loss')
axes[0].set_title('Функция потерь')
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Эпоха')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Точность')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# 5. СРАВНИТЕЛЬНЫЙ АНАЛИЗ
# ============================================================================
print("\n" + "="*60)
print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ МОДЕЛЕЙ")
print("="*60)

results = []
for name, history in histories.items():
    best_val_acc = max(history['val_accuracy'])
    best_epoch = history['val_accuracy'].index(best_val_acc) + 1
    final_val_acc = history['val_accuracy'][-1]
    final_train_acc = history['accuracy'][-1]
    overfitting = final_train_acc - final_val_acc
    
    results.append({
        'Модель': name.split(':')[0].strip(),
        'Лучшая val_accuracy': "{:.4f}".format(best_val_acc),
        'Final val_accuracy': "{:.4f}".format(final_val_acc),
        'Final train_accuracy': "{:.4f}".format(final_train_acc),
        'Переобучение': "{:.4f}".format(overfitting),
        'Эпоха лучшего результата': best_epoch,
        'Параметры': models[[m[0] for m in models].index(name)][1].count_params()
    })

results_df = pd.DataFrame(results)
print("\nСравнительная таблица результатов:")
print("-" * 80)
print(results_df.to_string(index=False))

# Определение лучшей модели
best_idx = np.argmax([max(h['val_accuracy']) for h in histories.values()])
best_model_name = list(histories.keys())[best_idx]
best_model_acc = max(histories[best_model_name]['val_accuracy'])

print("\n" + "="*60)
print("ВЫВОДЫ:")
print("="*60)
print("1. Лучшая модель: {}".format(best_model_name))
print("2. Лучшая точность на валидации: {:.4f}".format(best_model_acc))
print("3. Количество эпох обучения: {}".format(results_df.loc[best_idx, 'Эпоха лучшего результата']))
print("4. Все модели успешно обучены на 3 классах: rock, paper, scissors")