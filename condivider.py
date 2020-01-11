import io
import numpy as np
import random
import sys
import time
from keras.utils.data_utils import get_file
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
#scarico i dati
print("Download file...")
path = get_file('divina_commedia.txt', origin='https://www.retineuraliartificiali.net/keras_tutorial/divina_commedia.txt')
print("fatto!")
print("Apertura del file")
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
with open ('divina.commedia.txt','w') as f:
    f.write (text)
chars = sorted (list (set (text)))
char_indices = dict((c, i) for i, c in enumerate(chars)) # valore - numero
indices_char = dict((i, c) for i, c in enumerate(chars)) # numero - valore

maxlen = 30 # frazione di caratteri ridondante da me scelta
step = 3 # distanza tra i dati forniti alla rete
next_chars = []
sentences = [] #array dove metterò le frasi

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('numero frasi:', len(sentences))

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool) # matrice (amo le matrici) di input
y = np.zeros((len(sentences), len(chars)), dtype=np.bool) # output rappresentati ogni parola
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

model = Sequential() 
model.add(LSTM (128, input_shape=(maxlen, len(chars)))) #ottimizzata per gpu gentilemente offerrta da google perchè sono povero
model.add(Dense(len(chars), activation='softmax')) # normale layer di neuroni connessi completamente
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))


def testAfterEpoch (epoch,_):
    print()
    print()
    print('**************Epoch: %d**************' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    
  

    generated = ''
    sentence = text[start_index: start_index + maxlen] #crea un chunk di 30 caratteri a partire da un indice a caso
    generated += sentence
    print('*******************Frase di partenza*****************')
    print(sentence)
    print('*****************************************************')
    sys.stdout.write(generated)
    frase = generated +'\n'
    for i in range(400):
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
    model.save(f'{epoch}modello.h5')
    with open(str (epoch) + '.txt','w') as f:
        f.write (frase)
    print()

print_callback = LambdaCallback(on_epoch_end=testAfterEpoch)
#model.fit (x,y,batch_size=2048,epochs=65,callbacks=[print_callback])
