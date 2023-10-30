import tensorflow as tf
import numpy as np
from Entrainement import *

#Reseau de neurones
class OneHot(tf.keras.layers.Layer):
    def __init__(self, depth, **kwargs):
        super(OneHot, self).__init__(**kwargs)
        self.depth = depth

    def call(self, x):
        one_hot = tf.one_hot(tf.cast(x, tf.int32), self.depth)
        return one_hot

tf_inputs = tf.keras.Input(batch_shape=(4, 1))

one_hot = OneHot(4)(tf_inputs)

rnn_layer1 = tf.keras.layers.GRU(50, return_sequences=True, stateful=True)(one_hot)
rnn_layer2 = tf.keras.layers.GRU(50, return_sequences=True, stateful=True)(rnn_layer1)
hidden_layer = tf.keras.layers.Dense(50, activation="relu")(rnn_layer2)

out = tf.keras.layers.Dense(4, activation="softmax")(hidden_layer)

model = tf.keras.Model(inputs=tf_inputs, outputs=out)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

@tf.function
def train_step(inputs, targets):

    with tf.GradientTape() as tape:
        predictions = model(inputs)

        res = []
        for i in range(len(predictions)):
            res.append(predictions[i][0])
        res = tf.convert_to_tensor(res)

        loss = loss_object(targets, res)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(targets, res)

@tf.function
def predict(inputs_x):
    predictions = model(inputs_x)

    res = []
    for i in range(len(predictions)):
        res.append(tf.argmax(predictions[i][0]))
    print(predictions)
    return tf.convert_to_tensor(res)

model.reset_states()

def train() :
    for i in range(1) :
        game_data_X,game_data_Y = tf.convert_to_tensor(X[i]),tf.convert_to_tensor(Y[i])
        inputs,targets = game_data_X,game_data_Y

        for epoch in range(1000):
            for batch_inputs, batch_targets in zip(inputs, targets):
                batch_inputs = tf.reshape(batch_inputs, [-1])
                batch_targets= tf.reshape(batch_targets, [-1])
                train_step(batch_inputs, batch_targets)

            template = '\r Dim {}, Epoch {}, Train Loss: {}, Train Accuracy: {}'
            print(template.format(i+2, epoch+1,
                                  train_loss.result(),
                                  train_accuracy.result()*100), end="")
            model.reset_states()

    model.save('my_model.keras')

train()

model.load_weights('my_model.keras')
jeu = [[3,2],[0,2]]
board = tf.convert_to_tensor(jeu)
board = tf.reshape(board, [-1])

next_move = predict(board)

print("Le prochain coup est :",next_move)

