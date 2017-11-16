import warnings
from asl_data import SinglesData

def best_guess(logL):
    # base on the max value of all log likelihood
    # return the corresponding word
    return max(logL, key = logL.get)

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    probabilities = []
    guesses = []
    
    # Loop through all test set items and gather
    # thre log likelihood from the model
    for index in range(len(test_set.get_all_Xlengths())):
        # Get the features set and lengths for each test word
        word_X, word_lengths = test_set.get_item_Xlengths(index)
        
        logL = {}
        # Get the score for each model selected
        # by the model selector
        for word, model in models.items():
            try:
                score = model.score(word_X, word_lengths)
                # Construct the dictionary of
                # word : log likelihood
                logL[word] = score
            except:
                logL[word] = float("-inf")
                continue
        # Store the dictrionary into the list
        probabilities.append(logL)
        # Use the dictionary to find the best guess
        guesses.append(best_guess(logL))

    return probabilities,guesses
