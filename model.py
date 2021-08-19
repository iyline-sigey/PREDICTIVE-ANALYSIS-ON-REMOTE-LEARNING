import streamlit as st
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM, Bidirectional,Flatten,Dropout
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras import regularizers
from keras import layers
import re
import pickle


df=pd.read_csv('remote_clean.csv')
vocabulary_size = 10000
max_words = 5000
max_len = 200
#Neural Networks
st.header('Neural Networks')

X=df.clean_tweet.values
y=df.sentiment.values

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

    #  Split the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    # create the tokenizer that comes with Keras.
tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(X_train)

    #convert the texts to sequences.
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_test)

X_train_seq_padded = pad_sequences(X_train_seq, maxlen=200)
X_val_seq_padded  = pad_sequences(X_val_seq, maxlen=200)

#Intialize the model
model = Sequential()
model.add(layers.Embedding(max_words, 40, input_length=max_len))
model.add(layers.Bidirectional(layers.LSTM(20,dropout=0.6)))
model.add(layers.Dense(1,activation='sigmoid'))
        #Call comipiler ab=nd the checkpoints

model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])

        #fit the model

history = model.fit(X_train_seq_padded, y_train, epochs=10,validation_data=(X_val_seq_padded, y_test))
model.save('movie_sent.h5')

@st.cache
def predict(message):
    model=load_model('movie_sent.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        x_1 = tokenizer.texts_to_sequences([message])
        x_1 = pad_sequences(x_1, maxlen=500)
        prediction = model.predict(x_1)[0][0]
    return prediction

    
    
message = st.text_area("Enter Tweet,Type Here ..")
if st.button("Analyze"):
    with st.spinner("Analyzing the tweet …"):
        prediction=predict(message)
        if prediction >0.6:
            st.success("Positive review with {:.2f} confidence".format(prediction))
            st.balloons()
        elif prediction <0.40:
            st.error("Negative review with {:.2f} confidence".format(1-prediction))
        else:
            st.warning("Not sure! Try to add some more words")