import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import tensorflow as tf
import random
# import pymysql
from PIL import Image
import warnings
import re
import json
from PersonalData import EmpNoAndName

warnings.filterwarnings("ignore")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# db = pymysql.connect("localhost", "root", "526527492", "hrbot")
# cur = db.cursor()

bot_template = "BOT  :"

# Load trained data
with open("training_data.pkl", "rb") as file:
    data = pickle.load(file)
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# Load intents
with open('intents.json') as json_data:
    intents = json.load(json_data)

# Load trained model
model = tf.keras.models.load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

context = {}
ERROR_THRESHOLD = 0.25

def classify(sentence):
    results = model.predict(np.array([bow(sentence, words)]))[0]
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [(classes[r[0]], r[1]) for r in results]

def leave():
    print('BOT  : Please give your ID number to check your record')
    print("USER :", end=" ")
    while True:
        try:
            emp_id = int(input())
            # cur.execute("SELECT `leave`, Employee_Name FROM employees WHERE Employee_Number = %s", (emp_id,))
            # result = cur.fetchone()
            result = [23, 'John Doe']
            if not result:
                return print("BOT  : Please register on the panel first.")
            leave_balance, name = result
            name = name.replace(',', '')
            response_index = 0 if leave_balance > 0 else 1
            print(intents['intents'][6]["responses"][response_index].format(bot_template, name, leave_balance))
        except ValueError:
            print("BOT  : Please enter a valid Employee ID")
            print("USER :", end=" ")
            continue
        break

def policy():
    image = Image.open('policy.jpg')
    image.show()
    print("{} Please look at the image containing full policy details.".format(bot_template))

def find_name(message):
    name_keyword = re.compile(r'\b(name|call)\b', re.IGNORECASE)
    name_pattern = re.compile(r'\b[A-Z][a-z]*\b')
    if name_keyword.search(message):
        name_words = name_pattern.findall(message)
        if name_words:
            return ' '.join(name_words)
    return None

def recruitment():
    print(intents['intents'][3]["responses"][0].format(bot_template))
    print("USER :", end=" ")
    while True:
        name = input()
        name = find_name(name)
        if name:
            print("BOT  : Hello, {}! {}".format(name, intents['intents'][3]["responses"][1]))
            return
        print("{} Please use 'call me' or 'my name is' like: 'Call me John' OR 'My name is John'".format(bot_template))
        print("USER :", end=" ")

def response(sentence, userID='123'):
    results = classify(sentence)
    if not results:
        return print("BOT : Sorry, I didn't understand that.")
    for result in results:
        for intent in intents['intents']:
            if intent['tag'] == result[0]:
                if 'context_set' in intent:
                    context[userID] = intent['context_set']
                if 'context_filter' not in intent or (userID in context and intent['context_filter'] == context[userID]):
                    if result[0] == 'leave':
                        return leave()
                    if result[0] == 'policy':
                        return policy()
                    if result[0] == 'recruitment':
                        return recruitment()
                    if result[0] == 'EmpNoAndName':
                        return EmpNoAndName()
                    return print("{} {}".format(bot_template, random.choice(intent['responses'])))

while True:
    print("USER :", end=" ")
    message = input()
    if message.lower() == "bye":
        break
    response(message)