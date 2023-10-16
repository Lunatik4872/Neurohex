import tensorflow as tf
import numpy as np
from Entrainement import *

game_data_X,game_data_Y = np.array(X),np.array(Y)
inputs,targets = game_data_X,game_data_Y

#Reseau de neurones
class OneHot(tf.keras.layers.Layer) :
    def __init__(self, depth, **kwargs):
        super(OneHot, self).__init__(**kwargs)
        self.depth = depth

    def call(self, x, mask=None):
        x = tf.reshape(x, [-1]) # Reshape the input to be 1D
        one_hot = tf.one_hot(tf.cast(x, tf.int32), self.depth)
        return tf.reshape(one_hot, [-1, tf.shape(x)[-1], self.depth])

tf_inputs = tf.keras.Input(batch_shape=(64, 2))
one_hot = OneHot(20)(tf_inputs)

rnn_layer1 = tf.keras.layers.GRU(128, return_sequences=True, stateful=True)(one_hot)
rnn_layer2 = tf.keras.layers.GRU(128, return_sequences=True, stateful=True)(rnn_layer1)
hidden_layer = tf.keras.layers.Dense(128, activation="relu")(rnn_layer2)

out = tf.keras.layers.Dense(2, activation="softmax")(hidden_layer)

model = tf.keras.Model(inputs=tf_inputs, outputs=out)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

@tf.function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_object(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(targets, predictions)

@tf.function
def predict(inputs_x):
    predictions = model(inputs_x)
    predictions = predictions[0]

    return predictions

model.reset_states()
"""
for epoch in range(1000):
    for batch_inputs, batch_targets in zip(inputs, targets):

        batch_targets = tf.reshape(batch_targets, (1, -1))

        train_step(batch_inputs, batch_targets)

    template = '\r Epoch {}, Train Loss: {}, Train Accuracy: {}'
    print(template.format(epoch, train_loss.result(), train_accuracy.result()*100), end="")
    model.reset_states()

model.save('my_model.keras')"""

model.load_weights('my_model.keras')

board = np.array([[1,2], [-1,1]])
next_move = predict(board)
print(np.argmax(np.array(next_move)))
print("Le prochain coup est :", next_move)
