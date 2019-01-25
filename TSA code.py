# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Tweets.csv')

#Converting the sentiment to numerical format
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset['target'] = le.fit_transform(dataset['airline_sentiment'])
dataset.head(5)

#Eliminating all the irrelevant columns 
dataset = dataset.reindex(np.random.permutation(dataset.index)) 
dataset.reset_index(inplace=True)
dataset.drop('index',inplace=True,axis=1)
dataset = dataset[['text', 'target']]
dataset.head()

#Considering only positive & negative sentiments for analysis
dataset = dataset[dataset.target != 0]

# Cleaning the text
import re
dataset['text'] = dataset['text'].apply(lambda x: x.lower())
dataset['text'] = dataset['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

def remove_stopwords(self, input_text):
        stopwords_list = stopwords.words('english')
        # Some words which might indicate a certain sentiment are kept via a whitelist
        whitelist = ["n't", "not", "no"]
        words = input_text.split() 
        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
        return " ".join(clean_words)
    
def stemming(self, input_text):
        porter = PorterStemmer()
        words = input_text.split() 
        stemmed_words = [porter.stem(word) for word in words]
        return " ".join(stemmed_words)    
    
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(dataset['text'].values)
X = tokenizer.texts_to_sequences(dataset['text'].values)
X = pad_sequences(X)
y = dataset.iloc[:, 1].values

# Tfidf
from sklearn.feature_extraction.text import TfidfVectorizer # for vectorizing words 
from sklearn.model_selection import train_test_split
# vectorizing text
vectorizer = TfidfVectorizer('english')
text_vector = vectorizer.fit_transform(dataset)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, SpatialDropout1D
from keras.callbacks import ModelCheckpoint
import os
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Neural network
embed_dim = 128
lstm_out = 196
model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(Dropout(0.5))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


Y = pd.get_dummies(dataset['target']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

# Selecting some data for training & some for testing
X_val = X_train[:500]
Y_val = Y_train[:500]

partial_X_train = X_train[500:]
partial_Y_train = Y_train[500:]

# Train the network
batch_size = 512
history = model.fit(partial_X_train, 
                    partial_Y_train, 
                    epochs = 10, 
                    batch_size=batch_size, 
                    validation_data=(X_val, Y_val))


import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()






