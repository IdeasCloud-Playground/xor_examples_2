# from tensorflow.keras import Sequential
# from tensorflow.keras.layers.core import Dense, Dropout, Activation
# from tensorflow.keras import initializers
import numpy as np
import tensorflow as tf

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])


def initialize_weights(shape, dtype=None):
    return np.random.normal(loc=0.75, scale=1e-2, size=shape)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(2,
                                activation='sigmoid',
                                kernel_initializer=initialize_weights,
                                input_dim=2))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

print("*** Training... ***")

model.fit(X, y, batch_size=4, epochs=10000, verbose=0)

print("*** Training done! ***")

print("*** Model prediction on [[0,0],[0,1],[1,0],[1,1]] ***")

print(model.predict(X))
