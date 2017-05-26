import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # param series : the input data set 
    # param window_size : window size
    # output X,y: input/output pairs
    # containers for input/output pairs
    X = [] # initialize empty container to hold our inputs
    y = [] # initialize empty container to hold our outputs
    
    # For the series, loop through the series minus the window size.
    for i in range(window_size, len(series)):
        X.append(series[i - window_size:i])  # insert our inputs into the input array
        y.append(series[i]) # insert our output pairs into the output array
        
    # reshape each 
    X = np.asarray(X) # convert to a numpy array
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y) # convert to a numpy array
    y.shape = (len(y),1)
    
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):

    # given - fix random seed - so we can all reproduce the same results on our default time series
    np.random.seed(0)


    # TODO: build an RNN to perform regression on our time series input/output data
    model = Sequential() # initialize our model
    model.add(LSTM(5, input_shape=(window_size, 1))) # add a LSTM layer with 5 hidden units and shape of (window_size, 1)
    model.add(Dense(1, input_dim=window_size, activation='linear')) # add a linear activation layer with a single output to get our predicted price

    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)

### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # create a set of all unique characters in the text
    unique = set(text)
    ## print out our original set
    print(unique)

    # remove as many non-english characters and character sequences as you can 
    letters = 'abcdefghijklmnopqrstuvwxyz' # letters include all lowercase ascii letters
    punct = '!,.:;?' # eclamation mark, comma, period, colon, semicolon, question mark
    good_chars = set(letters + punct + ' ') # Creates a set including the letters, punctuation, and space

    for l in unique:  # loop through the set of characters in the text
        if not l in good_chars: # If this is a 'bad' and is not in our defined set of good character, replace it with a space!
            text = text.replace(l, ' ')

    # shorten any extra dead space created above
    text = text.replace('  ',' ')
    ## print out our new set!
    print(set(text))
    
### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # param text: input set
    # param window_size: window size to be used to transform the text
    # param step_size: size of step we want to iterate through for this text
    #output inputs,outputs: input/output pairs

    # containers for input/output pairs
    inputs = [] # Create empty array for inputs
    outputs = [] #create empty array to hold outputs
    
    # For each window, lets create an input set and output set.
    for i in range(window_size, len(text), step_size):   
        inputs.append(text[i - window_size:i])  #inputs is a set of characters the size of the window
        outputs.append(text[i])  # adds the next character in the text to the output
    
    return inputs,outputs