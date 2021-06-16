# Eccli Mario Stock price predict ai
Предсказание цен акции с помощью Python и Tensorflow (НЕ ЯВЛЯЕТСЯ ИНВЕСТИЦИОННОЙ РЕКОМЕНДАЦИЕЙ)

**Работа по ссылке -->** : <a href="https://colab.research.google.com/drive/1pEa9KaqEHggrNXWwxaayIBDrlJPuO5EG?usp=sharing">Google colab</a>

## Описание
Обучаем модель на историчесих данных, после чего тестируем и сравниваем с уже имеющимеся данными и в конце предсказываем цену акции на сл день. 
(НЕ ЯВЛЯЕТСЯ ИНВЕСТИЦИОННОЙ РЕКОМЕНДАЦИЕЙ)

# Загрузка данных 

```sh
# Выбираем тикер и указываем дату начала и конца исторических данных которых мы хотим получить.
company = 'BA'

start = dt.datetime(2012,1,1)
end = dt.datetime(2020,1,1)

data = dr.DataReader(company, 'yahoo', start, end)
```

# Подготовка данных

```sh
# Работаем с данными, делаем так, чтобы данные умещались между 0 и 1
scaler = MinMaxScaler(feature_range=(0,1))
```
Например: 
Минимальная цена - 10 долларов, то тут 0
...
...
Максимальная цена - 600 долларов, то тут 1 

```sh
# Мы заинтересованны только в цене закрытия, ибо будем предсказывать только цену после закрытия бирджи.
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
```

```sh
# сколько дней мы будем смотреть в прошлое, что бы предсказываать сл. день
prediction_days = 60
```

Подготовка данных для обучения 

```sh
x_train = []
y_train = []
# 60 значений имеем и еще одно, чтоб наша модель могла обучаться предсказывать сл значение
for x in range(prediction_days, len(scaled_data)):
  x_train.append(scaled_data[x-prediction_days:x, 0])
  y_train.append(scaled_data[x, 0])
```

Переводим в numpy array 
```sh
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
```
# Построение модели(слои)

```sh
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, 
               input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
```
# Обучение

25 раз в каждом разе по 32 единицы

```sh
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)
```

# Проверка на исторических данных

![Screenshot 2021-06-16 214129](https://user-images.githubusercontent.com/56271529/122274800-a709b600-ceeb-11eb-9280-fff2632308d6.png)


Алгоритм особо не переоценивает стоймость акции, но при этом довольно точно определяет, когда происходит рост и падение.
Стоит также учитывать, что мы смотрим назад на 60 дней, это довольно не много.

Можно также сделать так, чтоб он смотрел на свои данные(60 дней), которые он сам возвращает, но тогда результаты будут не такими точными.

# Выводим предсказание следущего дня
```sh
real_data = [model_inputs[len(model_inputs) + 1 #плюс один день
                         - prediction_days:len(model_inputs+1), 0]]
```

Используем настоящие данные на входе
```sh
prediction = model.predict(real_data)
```

