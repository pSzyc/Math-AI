import re 
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MathDataset(Dataset):

    def __init__(self, X, Y):
        self.n_samples = len(X)
        self.x_data = X
        self.y_data = Y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

def generate_data():

    train = open("train_data.txt", 'w')
    train.write('Expression,Value\n')
    
    test = open("test_data.txt", 'w')
    test.write('Expression,Value\n')
    
    array = np.random.choice([True, False], size=(100, 100, 4))

    for i in range(len(array)):
        for j in range(len(array[i])):
            for k in range(len(array[i][j])):

                a = i
                b = j

                temp = k
                if temp == 0:
                    operator = '+'
                elif temp == 1:
                    operator = '-'
                elif temp == 2:
                    operator = '*'
                elif temp == 3:
                    if(b == 0):
                        continue
                    operator = '/'

                expression = str(a) + ' ' + operator + ' ' + str(b)
                value = eval(expression)
                if array[i][j][k]:
                    test.write(expression+","+str(value) + '\n')
                else:
                    train.write(expression+","+str(value) + '\n')      
    train.close()
    test.close()

def converter(expression):
    '''Converts a string equation of type: a operator b to a vector.
    Vector should be read in the following way: [+,-,*,/, a value , b value] where +,-,*,/ are 0 or 1 depending on the presence (1) or absence (0) of a given operator'''
    
    sentence = re.split(' ', expression)
    vector = []
    vector.append(int(sentence[0]))
    
    for i in range(0, len(sentence)-1, 2):
        operator = sentence[i+1]
        number = int(sentence[i+2])
        
        if operator == '-' or operator == 'minus':
            vector.extend([1,0,0,0])
        elif operator == '+' or operator == 'plus':
            vector.extend([0,1,0,0])
        elif operator == '*' or operator == 'times':
            vector.extend([0,0,1,0])
        elif operator == '/' or operator == 'divided_by':
            vector.extend([0,0,0,1])
        else:
            raise ValueError("Inappropriate operator")
        vector.append(number)
    return vector

generate_data()