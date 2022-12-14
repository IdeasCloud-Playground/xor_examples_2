# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation
# from keras.optimizers import SGD
import numpy as np
import tensorflow as tf

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_dim=2),
    tf.keras.layers.Activation('tanh'),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Activation('sigmoid')
])
# model.add(tf.keras.layers.Dense(8, input_dim=2))
# model.add(tf.keras.layers.Activation('tanh'))
# model.add(tf.keras.layers.Dense(1))
# model.add(tf.keras.layers.Activation('sigmoid'))

sgd = tf.keras.optimizers.SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd)

model.fit(X, y,  batch_size=1, epochs=1000)

print(model.predict(X))
"""
[[ 0.0033028 ]
 [ 0.99581173]
 [ 0.99530098]
 [ 0.00564186]]
"""
