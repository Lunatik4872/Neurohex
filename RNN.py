import tensorflow as tf
import numpy as np
from Entrainement import *

#traitement des données 

game_data_X,game_data_Y = X,Y
#print("X =",game_data_X,"\n\n","Y =",game_data_Y)

inputs,targets = game_data_X,game_data_Y

def gen_batch(inputs, targets, batch_size=32):
    assert len(inputs) == len(targets)
    num_batches = len(inputs) // batch_size
    
    for i in range(num_batches):
        batch_inputs = np.array(inputs[i*batch_size: (i+1)*batch_size])
        batch_targets = np.array(targets[i*batch_size: (i+1)*batch_size])
        
        yield batch_inputs, batch_targets


#Reseau de neurones
class OneHot(tf.keras.layers.Layer) :
    def __init__(self, depth, **kwargs):
        super(OneHot, self).__init__(**kwargs)
        self.depth = depth
    
    def call(self, x, mask=None):
        return tf.one_hot(tf.cast(x, tf.int32), self.depth)

tf_inputs = tf.keras.Input(shape=(None, ), batch_size=64)
one_hot = OneHot(2)(tf_inputs)

rnn_layer1 = tf.keras.layers.GRU(128, return_sequences=True, stateful=True)(one_hot)
rnn_layer2 = tf.keras.layers.GRU(128, return_sequences=True, stateful=True)(rnn_layer1)
hidden_layer = tf.keras.layers.Dense(128, activation="relu")(rnn_layer2)

out = tf.keras.layers.Dense(2, activation="softmax", )(hidden_layer)

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
    return predictions

model.reset_states()
"""
for epoch in range(4000):
    for batch_inputs, batch_targets in gen_batch(inputs, targets):
        train_step(batch_inputs, batch_targets)
    template = '\r Epoch {}, Train Loss: {}, Train Accuracy: {}'
    print(template.format(epoch, train_loss.result(), train_accuracy.result()*100), end="")
    model.reset_states()

model.save('my_model.keras')
"""
model.load_weights('my_model.keras')
example = np.array([[1,1],[1,1]])
example_batch = np.expand_dims(example, axis=0)
prediction = model.predict(example_batch)

print(prediction)






