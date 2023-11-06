#fait pas gaffe à ce fichier il me permet de convertir des données redondantes 
l = [
        [[2,1,2],[2,1,2],[2,1,2]],
        [[2,1,2],[2,1,2],[2,1,2]],
        [[2,1,2],[2,1,2],[2,1,2]],
        [[2,1,2],[2,1,2],[2,1,2]],
        [[2,1,2],[2,1,2],[2,1,2]],
        [[2,1,2],[2,1,2],[2,1,2]]
    ]

l2 = [
        [1,0,0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0],
        [0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0],
        [0,0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,0,1]
    ]


for i in range(len(l)) :
    pos = l2[i].index(max(l2[i]))
    l[i][pos//3][pos%3] = 3

import copy

test = [[2,1,1,2],[2,1,1,2],[2,1,1,2],[2,1,1,2]]
res = []

for i in range(16) :
    if test[i//4][i%4] == 2:
        temp = copy.deepcopy(test)
        temp[i//4][i%4] = 3
        res.append(temp)

for matrix in res:
    print(str(matrix)+",")
