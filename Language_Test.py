import numpy as np
import pandas as pd

from MyMLP import MLP

np.random.seed(1)

log = open("language_test.txt", "w")
print("LANGUAGE TEST\n", file = log)

inputs = []
outputs = []
doutput = []
d= []
columns=["letter","x-box","y-box","width","height","onpix","x-bar","y-bar","x2bar","y2bar","xybar","x2ybr","xy2br","x-ege","xegvy","y-ege","yegvx"]

df=pd.read_csv("letters-recog.data", names=columns)
doutput=df["letter"]

print(doutput[1])


for i in range(len(doutput)):
    outputs.append(ord(str(doutput[i])))

inputs=df.drop(["letter"], axis=1)
inputs=np.array(inputs)

print(outputs[1])

no_inputs = 16
no_hidden = 10
no_outputs = 26

learning_rate = 0.1
max_epochs = 1000

mlp = MLP(no_inputs, no_hidden, no_outputs)
mlp.randomise()
print('\nMax Epoch:\n' + str(max_epochs), file=log)
print('\nLearning Rate:\n' + str(learning_rate), file=log)
print('\nBefore Training:\n', file=log)

for i in range(16000):
    mlp.forward(inputs[:len(inputs) -4000 ], True)
    print(('Target:\t %s  Output:\t %s') % (str(outputs[i]), str(mlp.o[i])), file=log)

print('\nTraining:\n', file=log)

for e in range(0, max_epochs):
    mlp.forward(inputs)
    for i in range(16000):
        index=inputs[i]-65
        error = mlp.backwards(inputs[i], outputs[i][index],sin=True)

    mlp.update_weights(learning_rate)

    if (i + 1) % (max_epochs / 20) == 0:
        print(' Error at Epoch:\t' + str(e + 1) + '\t\t  is \t\t' + str(error), file=log)

print('\n After Training :\n', file=log)

for i in range(len(inputs) - 4000, len(inputs)):
    mlp.forward(inputs[i], True)
    print(('Target:\t %s  Output:\t %s') % (str(outputs[1][i]), str(mlp.o[0])), file=log)

iteration=[1000]
learn_rate=[0.1]
#
# for i in range(len(iteration)):
#     print('\n-------------------------------------------------------------------' , file=log)
#     print('|                                                                    |', file=log)
#     print('----------------------------------------------------------------------\n', file=log)
#     for j in range(len(learn_rate)):
#
#         run_Languagetest(iteration[i],learn_rate[j])
#         print('\n-------------------------------------------------------------------\n', file=log)