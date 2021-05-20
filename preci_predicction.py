import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#importandig data
sales_df = pd.read_csv("Sales_data.csv")

#display
sns.scatterplot(sales_df['Temperature'], sales_df['Revenue'])

#creating training set
x_train = sales_df['Temperature']
y_train = sales_df['Revenue']

#creating model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units = 1, input_shape = [1]))

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss = 'mean_squared_error')

#training
epochs_hist = model.fit(x_train, y_train, epochs = 1000)

keys = epochs_hist.history.keys()

#training chart
plt.plot(epochs_hist.history['loss'])
plt.title('Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend(['Training Loss'])

weights = model.get_weights()

#prediction
Temp = 50
Revenue = model.predict([Temp])
print('The gain according to the neural network will be: ', Revenue)

#Prediction chart
plt.scatter(x_train, y_train, color = 'gray')
plt.plot(x_train, model.predict(x_train), color = 'red')
plt.ylabel('Gain [Dollars]')
plt.xlabel('Temperature [gCelsius]')
plt.title('Generated profit vs Temperature @ Ice cream company')