import tensorflow as tf
import numpy as np
from Entrainement import *

#Reseau de neurones
class OneHot(tf.keras.layers.Layer):
    #OneHot c'est l'encodage 1 parmi n consiste a encoder une variable a n états sur n bits dont un seul prend la valeur 1 (c'est question de proba)
    def __init__(self, depth, **kwargs):
        super(OneHot, self).__init__(**kwargs)
        self.depth = depth

    def call(self, x):
        one_hot = tf.one_hot(tf.cast(x, tf.int32), self.depth)
        return one_hot

tf_inputs = tf.keras.Input(shape=(None,), batch_size=1) #Couche d'entré flexible qui prend n'importe quel taille de donné (J'ai pas fini la flexibilité)

one_hot = OneHot(4)(tf_inputs) #Je sais pas encore c'est quoi le meilleur traitement entre le one hot et le embedding
#embedding = tf.keras.layers.Embedding(input_dim=4, output_dim=64)(tf_inputs) #Créer un vecteur de taille identique pour tous (Le 4 represente les 4 valeurs de sortie possible) 

rnn_layer1 = tf.keras.layers.LSTM(20, return_sequences=True, stateful=True)(one_hot) #Couche du réseau RNN capable de mémoriser les états afin de jouer en fonction de ce qui c'est passé
rnn_layer2 = tf.keras.layers.LSTM(20, return_sequences=True, stateful=True)(rnn_layer1) #Chaque nouvelle couche est relié à la précedente 
hidden_layer = tf.keras.layers.Dense(20, activation="relu")(rnn_layer2) #Couche classique avec comme comme focntion la relu(x) = max(0,x)
outputs = tf.keras.layers.Dense(4, activation="softmax")(hidden_layer) #sortie en proba softmax c'est comme la sigmoide mais en plus performant (Le 4 represente les 4 valeurs de sortie possible) 

model = tf.keras.Model(inputs=tf_inputs, outputs=outputs)

#C'est les composant qui optimise et entraine le réseau
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

@tf.function
def train_step(inputs, targets):
    #La fonction d'entrainement utilisant la déscente de gradiant 
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_object(targets, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables)) #Mise à jour des poids du réseau avec optimisation

    train_loss(loss)
    train_accuracy(targets, predictions)

@tf.function
def predict(inputs_x):
    #Fonction qui traite les entrée pour faire des probas des données lisibles 
    inputs_x = tf.convert_to_tensor(jeu)
    inputs_x = tf.reshape(inputs_x, (1,-1))
    predictions = model(inputs_x)

    res = []
    for i in range(predictions[0].shape[0]) :
        res.append(tf.argmax(predictions[0][i]))
    
    return tf.convert_to_tensor(res)

model.reset_states()

def train() :
    for i in range(len(X)) :
        inputs,targets = X[i],Y[i]

        for epoch in range(1000): #le nombre de neurones doit etre 10 et 90 j'ai pas assez testé pour une val exact
            for batch_inputs, batch_targets in zip(inputs, targets):
                #On converti les données en matrices tensor et on les aplati
                train_step(tf.reshape(tf.convert_to_tensor(batch_inputs),(1,-1)), tf.reshape(tf.convert_to_tensor(batch_targets),(1,-1))) 

            template = '\r Dim {}, Epoch {}, Train Loss: {}, Train Accuracy: {}'
            print(template.format(i+2, epoch+1,train_loss.result(),train_accuracy.result()*100), end="")
            model.reset_states()

    model.save('my_model.keras')

train() #Pense à enlever le com pour entrainer le réseau 

def IA_jouer(tab) :
    #Un traitement suplémentaire pour les sortir de la forme tensor qui est pas jolie à voir et assez discrete 
    jeu = predict(tab)
    dim = int(jeu.shape[0]**0.5)
    return np.reshape(jeu.numpy(),(dim,dim))

model.load_weights('my_model.keras') #C'est le fichier ou on stock le réseau RNN 
jeu = [[3,0,2],
       [0,3,3],
       [2,0,2]] #Aie de l'overfitting bon faut jouer avec le nombre de noeurones et le nombre d'iteration pour l'apprentissage (S'il recopie betement les données c'est qu'il est en surapprentissage en gros il ne décide plus par lui même il ce base uniquement sur les données)

next_move = IA_jouer(jeu)

<<<<<<< HEAD
print("Le prochain coup est :\n",next_move)
=======
print("Le prochain coup est :\n",next_move)
>>>>>>> af8acb75615db6917546036f31665a364c66a858
