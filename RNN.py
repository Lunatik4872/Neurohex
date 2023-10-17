import tensorflow as tf
import numpy as np
from Entrainement import *

game_data_X,game_data_Y = tf.convert_to_tensor(X),tf.convert_to_tensor(Y)
inputs,targets = game_data_X,game_data_Y

#Reseau de neurones
class OneHot(tf.keras.layers.Layer) :
    def __init__(self, depth, **kwargs):
        super(OneHot, self).__init__(**kwargs)
        self.depth = depth

    def call(self, x, mask=None):
        x = tf.reshape(x, (1,-1))
        one_hot = tf.one_hot(tf.cast(x, tf.int32), self.depth)
        return tf.reshape(one_hot, [-1, tf.shape(x)[-1], self.depth])

tf_inputs = tf.keras.Input(batch_shape=(64,))
one_hot = OneHot(4)(tf_inputs)

rnn_layer1 = tf.keras.layers.GRU(128, return_sequences=True, stateful=True)(one_hot)
rnn_layer2 = tf.keras.layers.GRU(128, return_sequences=True, stateful=True)(rnn_layer1)
hidden_layer = tf.keras.layers.Dense(128, activation="relu")(rnn_layer2)

out = tf.keras.layers.Dense(2,activation="softmax")(hidden_layer)#possible 4 au lieu de 2

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
    max_indices = tf.argmax(predictions, axis=-1)[0] #possible qu'il faudra le sortir de la 

    return max_indices

model.reset_states()

for epoch in range(4000):
    for batch_inputs, batch_targets in zip(inputs, targets):

        train_step(batch_inputs, batch_targets)

    template = '\r Epoch {}, Train Loss: {}, Train Accuracy: {}'
    print(template.format(epoch, train_loss.result(), train_accuracy.result()*100), end="")
    model.reset_states()

model.save('my_model.keras')

model.load_weights('my_model.keras')

board = tf.convert_to_tensor([[1,-1], [1,2]])

next_move = predict(board)

print("Le prochain coup est :", next_move)
