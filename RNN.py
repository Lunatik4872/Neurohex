import tensorflow as tf
import numpy as np
from Entrainement import *

#traitement des données 

game_data_X,game_data_Y = X,Y
#print("X =",game_data_X,"\n\n","Y =",game_data_Y)

inputs,targets = game_data_X,game_data_Y

def gen_batch(inputs, targets):
    assert len(inputs) == len(targets)
    num_batches = len(inputs) 
    
    for i in range(num_batches):
        batch_inputs = np.array(inputs[i: (i+1)]).flatten()
        batch_targets = np.array(targets[i : (i+1)]).flatten()
        
        yield batch_inputs, batch_targets

#for elt in range(len(inputs)) :
#    for batch_inputs, batch_targets in gen_batch(inputs[elt], targets[elt]):
#        print(batch_inputs[0], batch_targets[0])

#Reseau de neurones 
class OneHot(tf.keras.layers.Layer) :
    def __init__(self, depth, **kwargs):
        super(OneHot, self).__init__(**kwargs)
        self.depth = depth
    
    def call(self, x, mask=None):
        return tf.one_hot(tf.cast(x, tf.int64), self.depth)
    

tf_inputs = tf.keras.Input(shape=(None,), batch_size=64)
one_hot = OneHot(4)(tf_inputs)

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
def predict(inputs):
    predictions = model(inputs)
    return predictions

model.reset_states()

for i in range(100):
    for batch_inputs, batch_targets in gen_batch(inputs, targets):
        batch_inputs_flattened = np.array(batch_inputs)
        train_step(batch_inputs_flattened, batch_targets)
    template = '\r Iteration {}, Train Loss: {}, Train Accuracy: {}'
    print(template.format(i, train_loss.result(), train_accuracy.result()*100), end="")
    model.reset_states()

model.save("model_rnn.h5")




