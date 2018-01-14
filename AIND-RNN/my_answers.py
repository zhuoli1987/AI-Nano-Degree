import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras
import re


# DONE: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    # loop for spliting the series into input/output
    for index in range(len(series)-window_size):
        input_ = series[index:index+window_size]
        output_ = series[index+window_size]
        X.append(input_)
        y.append(output_)

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# DONE: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    lstm_size = 5
    
    # layer 1 uses an LSTM module with 5 hidden units (note here the input_shape = (window_size,1))
    model.add(LSTM(lstm_size, dropout=0.0, recurrent_dropout=0.0, input_shape=(window_size, 1)))
    
    # layer 2 uses a fully connected module with one unit
    model.add(Dense(1, activation=None))
    
    return model


### DONE: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    
    # remove special characters except for the once from the punctuation
    text = re.sub(r'[^a-zA-Z!,.:;?]', ' ',text)
    
    # shorten any extra dead space created above
    text = text.replace('  ',' ')
        
    return text

### DONE: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    # loop for spliting the series into input/output
    for index in range(0, len(text)-window_size, step_size):
        input_ = text[index:index+window_size]
        output_ = text[index+window_size]
        inputs.append(input_)
        outputs.append(output_)

    return inputs,outputs

# DONE build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    
    lstm_size = 200
    
    # layer 1 should be an LSTM module with 200 hidden units
    model.add(LSTM(lstm_size, dropout=0.2, recurrent_dropout=0.0, input_shape=(window_size, num_chars)))
    
    # layer 2 should be a linear module, fully connected, with len(chars) hidden units
    model.add(Dense(num_chars))
    
    # layer 3 should be a softmax activation 
    model.add(Activation("softmax"))
    
    return model
