## 30 задач по простой полносвязной нейронной сети с изображениями в Keras

### Задача 1: Подготовка данных
**Описание:** Загрузите и предобработайте данные (например, набор данных mnist) для использования в простой полносвязанной нейронной сети.

```python
# Код
from keras.datasets import mnist
from keras.utils import to_categorical

# Загрузка данных
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Предобработка данных
x_train = x_train.reshape((x_train.shape[0], -1)).astype('float32') / 255.0
x_test = x_test.reshape((x_test.shape[0], -1)).astype('float32') / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
```

##### Датасет MNIST

MNIST (Modified National Institute of Standards and Technology) - это набор данных, широко используемый в машинном обучении для обучения и тестирования моделей распознавания рукописных цифр. Датасет был создан, чтобы предоставить стандартизированный набор данных для оценки производительности различных алгоритмов машинного обучения в задачах распознавания цифр от 0 до 9.

- Структура Датасета:
  - **Обучающая выборка:** Включает 60,000 рукописных изображений цифр от 0 до 9, каждое размером 28x28 пикселей.
  - **Тестовая выборка:** Содержит 10,000 изображений для оценки производительности обученной модели.

- Использование в Примерах:
    - В примерах, связанных с нейронными сетями и машинным обучением, MNIST часто используется как стандартный набор данных для обучения моделей распознавания изображений. Каждое изображение представляет отдельную цифру, а цель состоит в том, чтобы обучить модель правильно классифицировать эти цифры.

### Задача 2: Создание Простой Сети
**Описание:** Создайте простую полносвязанную нейронную сеть с несколькими слоями.

```python
# Код
from keras.models import Sequential
from keras.layers import Dense

# Создание модели
model = Sequential()
model.add(Dense(128, input_shape=(784,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

`Sequential` - это линейный стек слоев нейронной сети в Keras. Это удобный способ построения моделей, когда информация проходит через слои от одного конца к другому. В простых случаях, когда вы хотите построить модель слой за слоем в последовательном порядке, `Sequential` - это удобный выбор.


### Задача 3: Компиляция Модели
**Описание:** Скомпилируйте созданную модель с выбранным оптимизатором и функцией потерь.

```python
# Код
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

##### `Среднеквадратичная ошибка (Mean Squared Error, MSE)`
Среднеквадратичная ошибка измеряет среднеквадратичное отклонение между фактическими и предсказанными значениями. Часто используется в задачах регрессии.

```python
model.compile(loss='mean_squared_error', optimizer='adam')
```

### `Кросс-энтропия (Cross-Entropy)`
Кросс-энтропия часто применяется в задачах классификации. Бинарная кросс-энтропия используется для двухклассовой классификации, а категориальная - для многоклассовой.

**Бинарная кросс-энтропия**:

```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```
**Категориальная кросс-энтропия**:

```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

#### **Информация о кастомной функции потерь**
Если вам нужно использовать свою собственную функцию потерь, вы можете определить ее и передать в параметр `loss` при компиляции модели.

```python
from keras import backend as K

def custom_loss(y_true, y_pred):
    # Ваша кастомная функция потерь
    return K.mean(K.square(y_true - y_pred), axis=-1)

model.compile(loss=custom_loss, optimizer='adam')
```

### Задача 4: Обучение Модели
**Описание:** Обучите модель на обучающем наборе данных.

```python
# Код
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
```

### Задача 5: Оценка Производительности
**Описание:** Оцените производительность обученной модели на тестовом наборе данных.

```python
# Код
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {test_acc}')
```
#### Оценка точности в Keras

`Accuracy (Точность)`

В Keras, "accuracy" (точность) представляет собой метрику, используемую для измерения процента правильных предсказаний модели на тестовом наборе данных. Это особенно важно в задачах классификации, где модель классифицирует входные данные на различные категории.

Точность рассчитывается как отношение числа правильных предсказаний к общему числу примеров в тестовом наборе. Высокое значение точности обычно указывает на хорошую способность модели к корректному предсказанию классов.

Точность является одним из ключевых показателей эффективности модели и часто используется при оценке ее производительности.

Пример использования в обучении:

```python
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
```

Для анализа процесса обучения и оценки качества модели в ходе тренировки, вы можете извлечь значение точности из `history`:

```python
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']
```

Эти значения могут быть использованы для построения графиков и оценки производительности модели во времени.

### Задача 6: Использование Обученной Модели
**Описание:** Используйте обученную модель для предсказания классов для новых изображений.

```python
# Код
predictions = model.predict(x_test[:5])
print(predictions)
```

### Задача 7: Изменение Архитектуры Сети
**Описание:** Измените архитектуру сети, добавив дополнительные слои или увеличив/уменьшив количество нейронов.

```python
# Код
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

### Задача 8: Использование Регуляризации
**Описание:** Добавьте L1 или L2 регуляризацию к слоям модели для предотвращения переобучения.

```python
# Код
from keras.regularizers import l2

model.add(Dense(128, input_shape=(784,), activation='relu', kernel_regularizer=l2(0.01)))
```
#### Регуляризация

В контексте нейронных сетей, регуляризация - это метод контроля за переобучением модели путем добавления штрафа к функции потерь на основе сложности модели. Это помогает улучшить обобщающую способность модели и снизить вероятность переобучения, особенно когда модель имеет большое количество параметров.

##### L1 и L2 Регуляризация

- **L1 регуляризация (Lasso)** добавляет к функции потерь абсолютное значение весов модели. Это может привести к разреженным весам, что означает, что некоторые веса становятся точно равными нулю.

- **L2 регуляризация (Ridge)** добавляет к функции потерь квадрат весов модели. Это штрафует за большие веса, но не делает их нулевыми.


### Задача 9: Использование Различных Оптимизаторов
**Описание:** Попробуйте различные оптимизаторы (Adam, SGD, RMSprop) и сравните их результаты.

```python
# Код
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Задача 10: Изменение Функции Активации
**Описание:** Измените функции активации в слоях модели (relu, sigmoid, tanh) и сравните их влияние на производительность.

```python
# Код
model.add(Dense(128, input_shape=(784,), activation='sigmoid'))
```

### Задача 11: Использование Дропаута
**Описание:** Добавьте слои дропаута для предотвращения переобучения модели.

```python
# Код
from keras.layers import Dropout

model.add(Dropout(0.3))
```

**Dropout** - это техника регуляризации, при которой случайно выбранные нейроны отключаются во время обучения. Это помогает предотвратить слишком сильную зависимость между определенными нейронами и способствует обучению более устойчивых и обобщающих моделей.


### Задача 12: Изменение Размера Пакета
**Описание:** Измените размер пакета обучения и сравните скорость сходимости.

```python
# Код
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
```

### Задача 13: Обучение на Нескольких Эпохах
**Описание:** Увеличьте количество эпох обучения и оцените, как это влияет на производительность модели.

```python
# Код
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
```

### Задача 14: Использование Колбэков
**Описание:** Используйте колбэки для сохранения лучшей модели во время обучения.

```python
# Код
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[checkpoint])
```

### Задача 15: Изменение Размера Нейронов в Скрытых Слоях
**Описание:** Экспериментируйте с различным количеством нейронов в скрытых слоях и оцените влияние на производительность.

```python
# Код
model.add(Dense(256, input_shape=(784,), activation='relu'))
```

### Задача 16: Использование Ранней Остановки
**Описание:** Примените раннюю остановку для прекращения обучения, если производительность на валидационном наборе перестала улучшаться.

```python
# Код
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
```

### Задача 17: Оценка Производительности на Тренировочном Наборе
**Описание:** Оцените производительность модели на тренировочном наборе и сравните ее с тестовым набором.

```python
# Код
train_loss, train_acc = model.evaluate(x_train, y_train)
print(f'Train Accuracy: {train_acc}')
```

### Задача 18: Использование Learning Rate Scheduler
**Описание:** Используйте планировщик learning rate для динамической регулировки скорости обучения во время обучения.

```python
# Код
from keras.callbacks import LearningRateScheduler

def lr_schedule(epoch):
    return 0.001 * (0.1 ** int(epoch / 10))

learning_rate_scheduler = LearningRateScheduler(lr_schedule)
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[learning_rate_scheduler])
```

### Задача 19: Использование Batch Normalization
**Описание:** Добавьте слои Batch Normalization для улучшения производительности сети.

```python
# Код
from keras.layers import BatchNormalization

model.add(BatchNormalization())
```

### Задача 20: Обучение на Дисбалансированных Классах
**Описание:** Если классы несбалансированы, примените взвешивание классов во время обучения.

```python
# Код
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y_train.argmax(axis=1)), y=y_train.argmax(axis=1))
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, class_weight=dict(enumerate(class_weights)))
```

### Задача 21: Использование Сверточных Слоев
**Описание:** Добавьте сверточные слои для обработки изображений.

```python
# Код
from keras.layers import Conv2D, Flatten

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
```
### Задача 22: Использование Augmentation для Улучшения Обучения
**Описание:** Примените техники аугментации данных (например, повороты, сдвиги, масштабирование) для улучшения обучения модели.

```python
# Код
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)
datagen.fit(x_train)

model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), epochs=10, validation_data=(x_test, y_test))
```

### Задача 23: Использование Transfer Learning
**Описание:** Примените технику Transfer Learning, используя предварительно обученную модель (например, VGG16, ResNet) для улучшения производительности на своем наборе данных.

```python
# Код
from keras.applications import VGG16
from keras.layers import GlobalAveragePooling2D

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

### Задача 24: Использование Многоклассовой Классификации
**Описание:** Расширьте модель для решения задачи многоклассовой классификации, например, классификации изображений на несколько категорий.

```python
# Код
from keras.datasets import cifar10
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

### Задача 25: Использование Регуляризации Dropout для Сверточных Слоев
**Описание:** Примените слои Dropout для сверточных слоев для предотвращения переобучения.

```python
# Код
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.25))
```

### Задача 26: Использование Различных Архитектур Сверточных Сетей
**Описание:** Экспериментируйте с различными архитектурами сверточных сетей (например, LeNet, AlexNet, ResNet) и оцените их производительность.

```python
# Код
from keras.applications import ResNet50

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
```

### Задача 27: Использование Learning Rate Finder
**Описание:** Примените метод "learning rate finder" для определения оптимальной скорости обучения.

```python
# Код
from keras_lr_finder import LRFinder

lr_finder = LRFinder(min_lr=1e-5, max_lr=1e-2, steps_per_epoch=len(x_train)//32, epochs=3)
model.fit(x_train, y_train, epochs=3, batch_size=32, callbacks=[lr_finder])
```

### Задача 28: Использование Автокодировщика для Предварительного Обучения
**Описание:** Используйте автокодировщик для предварительного обучения, затем передайте полученные веса в нейронную сеть для классификации.

```python
# Код
from keras.layers import Input, Dense
from keras.models import Model

input_layer = Input(shape=(784,))
encoded = Dense(256, activation='relu')(input_layer)
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=10, batch_size=32)

weights = autoencoder.get_weights()

classification_model = Sequential()
classification_model.add(Dense(128, input_shape=(784,), activation='relu', weights=weights[:2]))
```

### Задача 29: Использование Attention Mechanism
**Описание:** Реализуйте механизм внимания (attention mechanism) для улучшения работы сети.

```python
# Код
from keras.layers import Attention

attention = Attention()  # Добавьте слой внимания в модель
```

### Задача 30: Оптимизация Гиперпараметров
**Описание:** Используйте методы оптимизации гиперпараметров (например, Grid Search, Random Search) для поиска оптимальных значений параметров модели.

```python
# Код (пример с использованием Grid Search)
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

def create_model(optimizer='adam', activation='relu'):
    model = Sequential()
    model.add(Dense(128, input_shape=(784,), activation=activation))
    model.add(Dense(64, activation=activation))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32, verbose=0)
param_grid = {'optimizer': ['adam', 'sgd', 'rmsprop'], 'activation': ['relu', 'sigmoid', 'tanh']}
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(x_train, y_train)
```