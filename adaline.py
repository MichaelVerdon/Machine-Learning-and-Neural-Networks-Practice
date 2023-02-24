import random
#########################################################
#Train a single perceptron to recognise 2 matrix patterns
#Supervised learning
#########################################################
input1 = [
    [0, 1, 0], #Cross Pattern
    [1, 1, 1],
    [0, 1, 0]
]

input2 = [
    [0, 0, 0], #Line pattern
    [1, 1, 1],
    [0, 0, 0]
]

inputs = [
    [0,1,0,1,1,1,0,1,0], #Pattern 1, target:1
    [0,0,0,1,1,1,0,0,0], #Pattern 2, target:1
    [1,0,1,0,1,0,1,0,1], #Random, target:0
    [0,0,1,1,0,0,1,1,1], #Random, target:0
]
#########################################################
class Perceptron:

    def __init__(self, nin):
        self.weights = [random.uniform(-1,1) for i in range(len(nin[0]))]
        self.nin = nin
        self.bias = 1

    def __call__(self): 
        weightedSums = []
        for inputs in self.nin:
            weightedSums.append(sum((wi*xi for wi,xi in zip(self.weights,inputs)), self.bias))

        def binaryStep(sumList):
            outputList = []
            for sum in sumList:
                if(sum >= 0):
                    outputList.append(1)
                else:
                    outputList.append(0)
            return outputList
            
        return binaryStep(weightedSums)
    
        
    def getSumTable(self, index):
        return [x for x in self.nin[index]]
#########################################################
    
adaline = Perceptron(inputs)

targets = [1, 1, 0, 0]
learningRate = 0.25

print("Initial Weights are: {}".format(adaline.weights) + "\n")
print("Predictions are: {}".format(adaline()) + "\n")
print("Targets are: {}".format(targets) + "\n")

def epoch():

    errorList = []

    counter = 0

    for output,target in zip(adaline(),targets):
        errorList.append(computeErrorRow(counter,output,target))
        counter += 1

    weightChanges = [0]*len(adaline.weights)

    for i in range(len(errorList)):
        for error in errorList:
            weightChanges[i] += error[i]
    
    weightChanges = [i * learningRate for i in weightChanges]

    for i in range(len(adaline.weights)):
        adaline.weights[i] = adaline.weights[i] + weightChanges[i]


def binaryStep(sumList):
            outputList = []
            for sum in sumList:
                if(sum >= 0):
                    outputList.append(1)
                else:
                    outputList.append(0)
            return outputList

def computeErrorRow(index,output,target):
    errorRow = []
    sumTable = adaline.getSumTable(index)
    for i in range(len(adaline.weights)):
        if(output == 1 and target == 0):
            errorRow.append(-sumTable[i])
        elif(target == 1 and output == 0):
            errorRow.append(sumTable[i])
        else:
            errorRow.append(0)
    return errorRow
#########################################################
def main():
    epochs = 0
    results = [0,0,0,0]
    while results != targets:

        epoch()
        results = adaline()
        epochs += 1
        print("#################################################################")
        print("Result from epoch {}: {}".format(epochs, results))
        roundedWeights = [round(i,5) for i in adaline.weights]
        print("Weights = {}".format(roundedWeights))
    
    print("#################################################################")
    print("Took {} epochs".format(epochs))
    print("Final weights are: {}".format(roundedWeights))

main()
#########################################################