import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria:
    
        BIC = -2 * logL + p * logN
        
        - "L" is likelihood of "fitted" model
        - "p" is the number of free parameters in model, also called the complexity
        - "p * log N" is the "penalty term" (increases with higher "p"
          to penalise complexity and avoid overfitting)
        - "N" is qty of data points (size of data set)
    """
    def free_param(self, num_states, num_data_points):
        # p = num_free_params = ("transition P(x)" == n*n) + means(n*f) + covars(n*f)
        #                     = n * n + (n * f) + (n * f) - 1
        #  num_free_params: "number of parameters yet to be estimated"
        #  n: number of model states
        #  f: number of data points (aka features) used to train the model (i.e. len(self.X[0]))
        #  P(x): probability
        return (num_states ** 2) + (2 * num_states * num_data_points) - 1
    
    def bic_score(self, logL, p, logN):
        return (-2 * logL) + (p * logN)
    
    def best_bic_score(self, score_bics):
        # Select the min score, according to the BIC
        # lower score means better model
        return min(score_bics, key = lambda x : x[0])

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        bic_scores = []
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(num_states)
                # Calculate the log likelihood
                logL = model.score(self.X, self.lengths)
                # Get the number of data points
                num_data_points = sum(self.lengths)
                # Calculate the p (Number of free paramters)
                p = self.free_param(num_states, num_data_points)
                # Calculate the logN
                logN = np.log(num_data_points)
                # Calculate the BIC score
                score = self.bic_score(logL, p, logN)
                bic_scores.append(tuple([score, model]))
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, num_states))
                pass
        
        # Since bic_scores is tuple of [bic_score, hmmmodel]
        # we want to return the model as the result
        return self.best_bic_score(bic_scores)[1] if bic_scores else None

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    
        DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
        
        - "log(P(X(i))" is the log likelihood of the word i
        - "1/(M - 1)SUM" is the mean value
        - "log(P(X(all but i)" is the log lokelihood of all the competeing words
    '''
    def anti_logL(self, model_logL, compete_words):
        return [model_logL[0].score(word[0], word[1]) for word in compete_words]
    
    def best_dic_score(self, dic_scores):
        # Select the max score, according to the DIC
        # higher score means better model
        return max(dic_scores, key = lambda x : x[0])
    
    def dic_score(self, model_logL, compete_words):
        return model_logL[1] - np.mean(self.anti_logL(model_logL, compete_words))
    
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        dic_scores = []
        compete_words = []
        model_logLs = []
        
        # Construct the compete words list
        for word in self.words:
            if word != self.this_word:
                compete_words.append(self.hwords[word])
    
        # Calculate the likelihood of the word
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(num_states)
                # Calculate the log likelihood
                logL_word = model.score(self.X, self.lengths)
                # Store the model as well as logL in the list
                model_logLs.append(tuple([model, logL_word]))
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, num_states))
                pass
                    
        # Calculate the likelihood of competing words
        for model_logL in model_logLs:
            score = self.dic_score(model_logL, compete_words)
            dic_scores.append(tuple([score, model_logL[0]]))
        
        # Since dic_scores is tuple of [dic_scores, hmmmodel]
        # we want to return the model as the result
        return self.best_dic_score(dic_scores)[1] if dic_scores else None


class SelectorCV(ModelSelector):
    ''' Select best model based on average log Likelihood of cross-validation folds.
        Seperate the by folds and calculate the log likelihood for each fold and rotate.
        Finally, average all the results.
    '''
    def best_cv_score(self, cv_scores):
        # Select the max score, according to the DIC
        # higher score means better model
        return max(cv_scores, key = lambda x : x[0])

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        # Set the default fold value, n_split = 3
        kf = KFold()
        cv_scores = []
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                logLs = []
                model = self.base_model(num_states)
                
                # Check to make sure the sequence is long enough
                # to do the fold operation
                if len(self.sequences) > 2:
                    for cv_training_index, cv_test_index in kf.split(self.sequences):
                        # Set the features and lengths based on the fold result
                        # use combine_sequences to rearrange features
                        self.X, self.lengths = combine_sequences(cv_training_index, self.sequences)
                        # Set the test features and lengths for scoring
                        # use combine_sequence to rearrange features
                        X_test, lengths_test = combine_sequences(cv_training_index, self.sequences)
            
                        # Calculate the log likelihood
                        logL = model.score(X_test, lengths_test)
                        logLs.append(logL)
                else:
                    # Calculate the log likelihood
                    logL = model.score(self.X, self.lengths)
                    logLs.append(logL)
                
                # Find average log likelihood from all
                # folds
                score = np.mean(logLs)
                cv_scores.append(tuple([score, model]))
                    
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, num_states))
                pass

        # Since cv_scores is tuple of [cv_scores, hmmmodel]
        # we want to return the model as the result
        return self.best_cv_score(cv_scores)[1] if cv_scores else None
