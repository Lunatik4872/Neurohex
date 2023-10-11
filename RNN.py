import tensorflow as tf
import numpy as np
import ast

with open('Entrainement.txt', 'r') as f:
    data = f.read()

data = ast.literal_eval(data)
game_data_X,game_data_Y = data[0],data[1]
#print("X =",game_data_X,"\n\n","Y =",game_data_Y)

inputs,targets = game_data_X,game_data_Y

def gen_batch(inputs, targets):
    assert len(inputs) == len(targets)
    num_batches = len(inputs) 
    
    for i in range(num_batches):
        batch_inputs = inputs[i: (i+1)]
        batch_targets = targets[i : (i+1)]
        
        yield batch_inputs, batch_targets

#for elt in range(len(inputs)) :
#    for batch_inputs, batch_targets in gen_batch(inputs[elt], targets[elt]):
#        print(batch_inputs[0], batch_targets[0])





