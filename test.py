import numpy as np

# Le tableau original
original_arrays = [
    [[1,2], [-1,1]],
    [[-1,1], [1,2]],
    [[2,-1], [1,1]],
]

# Nombre d'exemplaires que vous voulez créer pour chaque tableau
num_copies = 10

# Créer une liste pour stocker les copies
copies = []

for original_array in original_arrays:
    for _ in range(num_copies):
        # Créer une copie du tableau original
        copy = np.copy(original_array)
        # Mélanger la copie de manière aléatoire
        np.random.shuffle(copy.ravel())
        # Ajouter la copie à la liste des copies
        copies.append(copy)

# Mélanger les copies
np.random.shuffle(copies)

# Convertir la liste des copies en un tableau numpy 3D
copies = np.array(copies)

# Afficher le tableau des copies
print(list(copies))
