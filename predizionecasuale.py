import io
import numpy as np
import random
import sys
import time
from keras.models import load_model
from keras.utils.data_utils import get_file
print ('linrerie importate')
#scarico i dati
print("Download file...")
path = get_file('divina_commedia.txt', origin='https://www.retineuraliartificiali.net/keras_tutorial/divina_commedia.txt')
print("fatto!")
print("Apertura del file")
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
print('Lunghezza del testo:', len(text))
print (text [-31:])
maxlen = 30 # frazione di caratteri ridondante da me scelta
step = 3 # distanza tra i dati forniti alla rete
next_chars = []
sentences = []


chars = sorted (list (set (text)))
char_indices = dict((c, i) for i, c in enumerate(chars)) # valore - numero
indices_char = dict((i, c) for i, c in enumerate(chars)) # numero - valore

def random_prediction (lunghezza,model):
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated = ''
    sentence = text[start_index: start_index + maxlen] #crea un chunk di 30 caratteri a partire da un indice a caso
    generated += sentence
    print('*******************Frase di partenza*****************')
    print(sentence)
    print('*****************************************************')
    sys.stdout.write(generated)
    frase = generated +'\n'
    for _ in range(lunghezza):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = np.argmax(preds)
        next_char = indices_char[next_index]

        sentence = sentence[1:] + next_char
        frase = frase + next_char
        sys.stdout.write(next_char)
        sys.stdout.flush()
    print ()


def trymodel (name,lunghezza = 400,times = 1):
    new_model = load_model (name)
    for _ in range (times):
        random_prediction (lunghezza,new_model)

def predictd_dante (file_frase,model):
    with open (file_frase) as f:
        sentence = f.read ()
    if len (sentence) < 30:
        raise ValueError(f'La lunghezza della frase deve essere 30 invece di {len (sentence)} <Ruggiero Alessandro>')
    if type (model) is str:
        model = load_model (model)
    sys.stdout.write(sentence)
    sentence = sentence [-30:]
    for _ in range(1000):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = np.argmax(preds)
        next_char = indices_char[next_index]

        sentence = sentence[1:] + next_char
        sys.stdout.write(next_char)
        sys.stdout.flush()
    print ()





try:
    if input ():
        while True:
            name = input ('nome:')
            chars_user = int (input ('caratteri:'))
            times = int (input ('numero tentativi:'))
            trymodel (name,chars_user,times)
    else:
        while True:
            predictd_dante ('fraseiniziale.txt','final.h5')
            input ()
except KeyboardInterrupt:
    print ('chiudo')