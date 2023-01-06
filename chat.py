import random
import json
import re
import torch

from bot_model import NeuralNet
from nltk_utils import bag_of_words, tokenize

from math_model import NeuralNet as M_model
from prepare_data import converter, MathDataset

def equation_extraction(message):
    
    message = message.replace("plus", "+")
    message = message.replace("minus", "-")
    message = message.replace("devide", "/")
    message = message.replace("multiply", "*")
    message = message.replace(" ", "")
    regex = r"(-?[0-9]*[.]?[0-9]*)(\+|-|\*|\/)(-?[0-9]*[.]?[0-9]*)"
    result = re.search(regex, message)
    try:
        a = result.group(1)
        operator = result.group(2)
        b = result.group(3)
        if operator != "/":
            flag = True
        else:
            flag = False
    
        return (a + " " + operator + " " + b, flag)
    except:
        return "Couldn't interpret the expression"



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

# load math_model

m_model = M_model(input_size=6, hidden_size=100, output_size=1)
m_model.load_state_dict(torch.load("learned20000.pth"))
m_model.eval()

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if tag == "funny":
                    if len(intent['responses']) != 0:
                        response = random.choice(intent['responses'])
                        intents['intents'][-1]['responses'].remove(response)
                        return response
                    else:
                        return "These were all jokes I know :("
                if tag == "Maths":
                    sentence, flag = equation_extraction(msg)
                    exp_vector = torch.tensor(converter(sentence), dtype=torch.float32)
                    temp = m_model(exp_vector)
                    if flag:
                        return int(temp.item())
                    return temp.item()
                return random.choice(intent['responses'])
    
    return "I do not understand..."


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)

