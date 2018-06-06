import numpy as np
from MyMLP import MLP

log = open("sintest3.txt", "w")
print("SINE TEST\n", file = log)

def run_SINTEST(max_epochs, learning_rate, no_hidden):
    np.random.seed(1)

    inputs = []
    outputs = []
    for i in range(0, 50):
        four_inputs_vector = list(np.random.uniform(-1.0, 1.0, 4))
        four_inputs_vector = [float(four_inputs_vector[0]), float(four_inputs_vector[1]), float(four_inputs_vector[2]),
                              float(four_inputs_vector[3])]
        inputs.append(four_inputs_vector)

    inputs = np.array(inputs)

    for i in range(0, 50):
        outputs.append([inputs[i][0] - inputs[i][1] + inputs[i][2] - inputs[i][3]])

    no_inputs = 4
    no_outputs = 1
    mlp = MLP(no_inputs, no_hidden, no_outputs)
    mlp.randomise()
    print('\nMax Epoch:\n' + str(max_epochs), file=log)
    print('\nLearning Rate:\n' + str(learning_rate), file=log)
    print('\nBefore Training:\n', file=log)

    for i in range(0, 40):
        mlp.forward(inputs[i], sin=True)
        print(('Target:\t %s  Output:\t %s') % (str(outputs[i]), str(mlp.o)), file=log)

    print('Training:\n', file=log)


    for i in range(0, max_epochs):
        error = 0
        mlp.forward(inputs[:len(inputs) - 10], True)

        error = mlp.backwards(inputs[:(len(inputs) - 10)], outputs[:len(inputs) - 10],
                              True)
        mlp.update_weights(learning_rate)
       #prints error every 5% of epochs
        if (i + 1) % (max_epochs / 20) == 0:
            print(' Error at Epoch:\t' + str(i + 1) + '\t\t  is \t\t' + str(error), file=log)

    print('\n After Training :\n', file=log)
    for i in range(len(inputs) - 10, len(inputs)):
        mlp.forward(inputs[i], sin=True)
        print(('Target:\t %s  Output:\t %s') % (str(outputs[i]), str(mlp.o)), file=log)



iteration=[10000000]
learn_rate=[0.1,0.01,0.006]

for i in range(len(iteration)):


    for j in range(len(learn_rate)):
         print('----------------------------------------------------------------------\n', file=log)
         run_SINTEST(iteration[i], learn_rate[j], no_hidden=10)
         print('\n-------------------------------------------------------------------\n', file=log)