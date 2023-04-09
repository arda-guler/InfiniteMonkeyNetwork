import random
import math

euler = 2.71828182845904523536028747135266249775724709369995 # good enough

def sigmoid(x):
    try:
        return (1 + euler**(-x))**(-1)
    except OverflowError:
        return 2**16

class Layer:
    def __init__(self, n_s, n_o):
        self.n_s = n_s
        self.n_o = n_o
        self.weights = [[1] * n_o] * n_s
        self.biases = [0] * n_s

    def forward(self, inputs):
        self.inputs = inputs
        outputs = []

        for idx_o in range(self.n_o):
            output = 0
            for idx_w in range(self.n_s):
                output += inputs[idx_w] * self.weights[idx_w][idx_o] + self.biases[idx_w]

            output = sigmoid(output)
            outputs.append(output)

        self.outputs = outputs
        return self.outputs

class IMNet:
    def __init__(self, layers):
        self.layers = layers
        self.outlayer = self.layers[-1]
        self.inlayer = self.layers[0]

    def fwd_cycle(self, inputs):
        self.inputs = inputs

        c_in = inputs
        for idx_l, layer in enumerate(self.layers):
            c_in = layer.forward(c_in)

        return self.outlayer.outputs

    def calc_error(self, target):
        error = 0
        for idx_o, o in enumerate(self.outlayer.outputs):
            error += (o - target[idx_o])**2

        self.target = target
        self.error = error
        return error

    def update(self, rate=1):
        deltas = []
        delta_biases = []

        for idx0, layer in enumerate(self.layers):
            deltas.append([])
            delta_biases.append([])
            for idx1 in range(layer.n_s):
                deltas[idx0].append([])
                for idx2 in range(layer.n_o):
                    deltas[idx0][idx1].append(random.uniform(-1, 1) * rate)

                delta_biases[idx0].append(random.uniform(-0.1, 0.1) * rate)

        self.deltas = deltas
        self.delta_biases = delta_biases

        for idx0, layer in enumerate(self.layers):
            for idx1 in range(layer.n_s):
                for idx2 in range(layer.n_o):
                    layer.weights[idx1][idx2] += deltas[idx0][idx1][idx2]
                    layer.weights[idx1][idx2] = max(min(layer.weights[idx1][idx2], 10), -10)

                layer.biases[idx1] += delta_biases[idx0][idx1]

    def reupdate(self, rate=0.2):
        for idx0, layer in enumerate(self.layers):
            for idx1 in range(layer.n_s):
                for idx2 in range(layer.n_o):
                    layer.weights[idx1][idx2] += self.deltas[idx0][idx1][idx2] * rate
                    layer.weights[idx1][idx2] = max(min(layer.weights[idx1][idx2], 10), -10)

                layer.biases[idx1] += self.delta_biases[idx0][idx1] * rate * 0.1

    def decw(self, rate):
        for layer in self.layers:
            for idx_o, o in enumerate(layer.outputs):
                if o > 0.5:
                    for idx_w in range(layer.n_s):
                        layer.weights[idx_w][idx_o] -= rate
                else:
                    for idx_w in range(layer.n_s):
                        layer.weights[idx_w][idx_o] += rate

    def __repr__(self):
        output = "----NETWORK----"
        for l in self.layers:
            output += "\n----NEWLAYER---\n"
            output += str(l.weights) + "\n"
            output += str(l.biases)

        output += "\n----ENDNET----\n"
        return output

def xor(x, y):
    if x and y:
        return 0

    if (not x) and (not y):
        return 0

    return 1

def generate_datasets(n):
    input_set = []
    target_set = []
    for i in range(n):
        i1 = random.randint(0, 1)
        i2 = random.randint(0, 1)
        input_set.append([i1, i2])

        target_set.append([xor(i1, i2)])

    return input_set, target_set

def generate_network():
    l0 = Layer(2, 2)
    l1 = Layer(2, 1)
    layers = [l0, l1]
    net = IMNet(layers)

    return net

def train(net, n_dataset, tolerance, max_cycle=1e6):
    inputs, targets = generate_datasets(n_dataset)

    cycle = 0
    errorsum = tolerance + 1000 * n_dataset
    old_errorsum = errorsum
    old_layers = net.layers.copy()
    urate = 1
    while errorsum > tolerance and cycle < max_cycle:

        net.update(urate)
        
        errorsum = 0
        for i in range(n_dataset):
            net.fwd_cycle(inputs[i])
            error = net.calc_error(targets[i])
            errorsum += error

        errorsum = errorsum / (n_dataset * len(targets[0]))
        amp = (1 - errorsum)

        print("Cycle:", cycle, "AMP:", round(amp, 4))

        cycle += 1

    # print(net)
    return net, inputs, targets

def main():
    net = generate_network()
    net, inputs, targets = train(net, 500, 0.1, 5000)

    print(net)
    
    print(inputs[0])
    print(targets[0])
    prediction = net.fwd_cycle(inputs[0])
    print(prediction)
    print("")

    print(inputs[1])
    print(targets[1])
    prediction = net.fwd_cycle(inputs[1])
    print(prediction)
    print("")

    print(inputs[2])
    print(targets[2])
    prediction = net.fwd_cycle(inputs[2])
    print(prediction)
    print("")

    f = open("weights.txt", "w")
    f.write(str(net))
    f.close()

    return net

net = main()
