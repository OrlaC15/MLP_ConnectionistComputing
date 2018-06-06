import numpy as np
from MyMLP import MLP

log = open("xortest.txt", "w")
print("XOR TEST\n", file = log)


def run_Xortest(max_epochs, learning_rate):


    np.random.seed(1)
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    outputs = np.array([[0], [1], [1], [0]])

    no_inputs = 2
    no_hidden = 2
    no_outputs = 1
    mlp = MLP(no_inputs, no_hidden, no_outputs)


    mlp.randomise()
    print('\nMax Epoch:\n' + str(max_epochs), file=log)
    print('\nLearning Rate:\n' + str(learning_rate), file=log)
    print('\nBefore Training:\n', file=log)
    for i in range(len(inputs)):
        mlp.forward(inputs[i])
        print(('Target:\t %s  Output:\t %s') % (str(outputs[i]), str(mlp.o)), file=log)

    print('\nTraining:\n', file=log)

    for i in range(0, max_epochs):
        mlp.forward(inputs)
        error = mlp.backwards(inputs, outputs)
        mlp.update_weights(learning_rate)

        if (i + 1) % (max_epochs / 20) == 0:
            print(' Error at Epoch:\t' + str(i + 1) + '\t\t  is \t\t' + str(error), file=log)

    print('\n After Training :\n', file=log)

    for i in range(len(inputs)):
        mlp.forward(inputs[i])
        print(('Target:\t %s  Output:\t %s') % (str(outputs[i]), str(mlp.o)), file=log)


iteration=[1000000,100000]
learn_rate=[1.0,0.8,0.6,0.4,0.2,0.02]

for i in range(len(iteration)):

    for j in range(len(learn_rate)):
        print('----------------------------------------------------------------------\n', file=log)
        run_Xortest(iteration[i],learn_rate[j])
        print('\n-------------------------------------------------------------------\n', file=log)