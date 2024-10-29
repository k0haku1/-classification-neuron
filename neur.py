from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np


dataset_dir = "C:/flowers"

# Загрузка обучающих данных
train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir + '/train',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256, 256)
)

# Загрузка валидационных данных
val_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir + '/val',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256, 256)
)

# Определение аугментации
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

# Применение аугментации к обучающему набору данных
train_dataset = train_dataset.map(
    lambda x, y: (data_augmentation(x, training=True), y)
)

# Кеширование и предвыборка данных
train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

model = models.Sequential()

# Первый сверточный блок
model.add(layers.Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(0.001), input_shape=(256, 256, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

# Второй сверточный блок
model.add(layers.Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))

# Третий сверточный блок
model.add(layers.Conv2D(128, (3, 3), kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.4))

# Четвертый сверточный блок (новый)
model.add(layers.Conv2D(256, (3, 3), kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.4))

# Глобальный пуллинг
model.add(layers.GlobalAveragePooling2D())

# Полносвязный слой
model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))

# Выходной слой с количеством классов, равным количеству категорий
model.add(layers.Dense(5, activation='softmax'))

# Выводим архитектуру модели
model.summary()

# Компиляция модели
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=15
)

test_loss, test_acc = model.evaluate(val_dataset)
print(f"Точность на тестовых данных: {test_acc * 100:.2f}%")

# Построение графиков потерь и точности
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(15)

plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Точность на обучении')
plt.plot(epochs_range, val_acc, label='Точность на валидации')
plt.legend(loc='lower right')
plt.title('Точность')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Потери на обучении')
plt.plot(epochs_range, val_loss, label='Потери на валидации')
plt.legend(loc='upper right')
plt.title('Потери')
plt.show()