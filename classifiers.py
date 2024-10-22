import numpy as np


# You need to build your own model here instead of using existing Python
# packages such as sklearn!


## But you may want to try these for comparison, that's fine.
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression

class BinaryClassifier(object):
    """Base class for classifiers.
    """
    def __init__(self):
        pass
    def fit(self, X, Y):
        """Train your model based on training set
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N
              is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where
              N is the number of sentences
        """
        pass
    def predict(self, X):
        """Predict labels based on your trained model
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N
            is the number of sentences, D is the size of feature dimensions
        
        Returns:
            array -- predict labels, such as an N shape array, where N is the
            number of sentences
        """
        pass


class AlwaysPredictZero(BinaryClassifier):
    """Always predict the 0
    """
    def predict(self, X):
        return [0]*len(X)

# TODO: Implement this
import numpy as np

class NaiveBayesClassifier(BinaryClassifier):
    """Enhanced Naive Bayes Classifier with TF-IDF, smoothing, and feature filtering."""
    def __init__(self, smoothing_factor=0.5, min_freq=5, max_freq_ratio=0.9):
        self.smoothing_factor = smoothing_factor
        self.pos_word_counts = {}
        self.neg_word_counts = {}
        self.total_pos_words = 0
        self.total_neg_words = 0
        self.vocab = set()
        self.num_pos_reviews = 0
        self.num_neg_reviews = 0
        self.prior_pos = 0
        self.prior_neg = 0
        self.idf = {}
        self.min_freq = min_freq
        self.max_freq_ratio = max_freq_ratio

    def compute_tf_idf(self, X):
        """Compute TF-IDF for input features with scaling."""
        num_docs = len(X)
        word_doc_freq = {}

        # Calculate document frequency for each term
        for features in X:
            for i, count in enumerate(features):
                if count > 0:
                    word_doc_freq[i] = word_doc_freq.get(i, 0) + 1

        # Calculate Inverse Document Frequency (IDF)
        self.idf = {word: np.log(num_docs / (1 + freq)) for word, freq in word_doc_freq.items()}

        # Calculate TF-IDF for each document
        tf_idf_X = []
        for features in X:
            tf_idf_features = np.array([features[i] * self.idf.get(i, 0) for i in range(len(features))])
            tf_idf_features = tf_idf_features / np.max(tf_idf_features)  # Scaling TF-IDF values
            tf_idf_X.append(tf_idf_features)

        return tf_idf_X

    def filter_features(self, X):
        """Filter features based on frequency constraints."""
        num_docs = len(X)
        word_freq = np.sum(X > 0, axis=0)

        # Filter by min and max frequency ratio
        selected_indices = [i for i, freq in enumerate(word_freq)
                            if self.min_freq <= freq <= self.max_freq_ratio * num_docs]

        filtered_X = np.array([[features[i] for i in selected_indices] for features in X])
        return filtered_X, selected_indices

    def fit(self, X, Y):
        """Train the Naive Bayes classifier with TF-IDF transformation and filtering."""
        # Apply feature filtering
        X, self.selected_indices = self.filter_features(X)

        # Apply TF-IDF transformation
        X = self.compute_tf_idf(X)

        self.num_pos_reviews = sum(Y)
        self.num_neg_reviews = len(Y) - self.num_pos_reviews

        # Calculate prior probabilities
        self.prior_pos = self.num_pos_reviews / len(Y)
        self.prior_neg = self.num_neg_reviews / len(Y)

        # Calculate word counts for positive and negative classes
        for features, label in zip(X, Y):
            if label == 1:
                self.total_pos_words += sum(features)
                for i, count in enumerate(features):
                    self.pos_word_counts[i] = self.pos_word_counts.get(i, 0) + count
            else:
                self.total_neg_words += sum(features)
                for i, count in enumerate(features):
                    self.neg_word_counts[i] = self.neg_word_counts.get(i, 0) + count

        # Apply Add-1 smoothing
        vocab_size = len(X[0])
        for i in range(vocab_size):
            self.pos_word_counts[i] = self.pos_word_counts.get(i, 0) + self.smoothing_factor
            self.neg_word_counts[i] = self.neg_word_counts.get(i, 0) + self.smoothing_factor

        self.total_pos_words += self.smoothing_factor * vocab_size
        self.total_neg_words += self.smoothing_factor * vocab_size

    def predict(self, X):
        """Predict labels for test data."""
        # Apply feature filtering and TF-IDF transformation
        X = np.array([[features[i] for i in self.selected_indices] for features in X])
        X = self.compute_tf_idf(X)

        predictions = []
        for features in X:
            pos_prob = np.log(self.prior_pos)
            neg_prob = np.log(self.prior_neg)

            for i, count in enumerate(features):
                if count > 0:
                    pos_prob += count * np.log(self.pos_word_counts.get(i, 1) / self.total_pos_words)
                    neg_prob += count * np.log(self.neg_word_counts.get(i, 1) / self.total_neg_words)

            predictions.append(1 if pos_prob > neg_prob else 0)

        return predictions







# citation: https://www.geeksforgeeks.org/implementation-of-logistic-regression-from-scratch-using-python/
# TODO: Implement this
class LogisticRegressionClassifier(BinaryClassifier):
    """Logistic Regression Classifier
    """
    def __init__( self, learning_rate=0.01, iterations=1000, regularization_strength=1) :         
        self.learning_rate = learning_rate         
        self.iterations = iterations 
        self.lambda_ = regularization_strength
          

    def fit( self, X, Y ) :         
        self.m, self.n = X.shape         
        self.W = np.zeros( self.n )         
        self.b = 0        
        self.X = X         
        self.Y = Y 
                            
        for i in range( self.iterations ) :             
            self.update_weights()             
        return self
        
    def update_weights( self ) :            
        A = 1 / ( 1 + np.exp( - ( self.X.dot( self.W ) + self.b ) ) ) 
          
        tmp = ( A - self.Y.T )         
        tmp = np.reshape( tmp, self.m )         
        dW = (np.dot(self.X.T, tmp) / self.m) + (self.lambda_ * self.W / self.m)       
        db = np.sum( tmp ) / self.m  
          
        self.W = self.W - self.learning_rate * dW     
        self.b = self.b - self.learning_rate * db 
          
        return self
    
    def predict( self, X ) :     
        Z = 1 / ( 1 + np.exp( - ( X.dot( self.W ) + self.b ) ) )         
        Y = np.where( Z > 0.5, 1, 0 )         
        return Y 


#Results before adding L2 regularization
# ===== Train Accuracy =====
# Accuracy: 1280 / 1400 = 0.9143 
# ===== Test Accuracy =====
# Accuracy: 208 / 250 = 0.8320 
# Time for training and test: 41.41 seconds


# lambda = 0.0001
# ===== Train Accuracy =====
# Accuracy: 1280 / 1400 = 0.9143 
# ===== Test Accuracy =====
# Accuracy: 208 / 250 = 0.8320 
# Time for training and test: 49.34 seconds

#lamda = 0.001
# ===== Train Accuracy =====
# Accuracy: 1280 / 1400 = 0.9143 
# ===== Test Accuracy =====
# Accuracy: 208 / 250 = 0.8320 
# Time for training and test: 38.34 seconds

# lamda = 0.01
# ===== Train Accuracy =====
# Accuracy: 1365 / 1400 = 0.9750 
# ===== Test Accuracy =====
# Accuracy: 198 / 250 = 0.7920 
# Time for training and test: 10.41 seconds

# lamda = 0.1
# ===== Train Accuracy =====
# Accuracy: 1280 / 1400 = 0.9143 
# ===== Test Accuracy =====
# Accuracy: 208 / 250 = 0.8320 
# Time for training and test: 40.64 seconds

# lambda = 1
# ===== Train Accuracy =====
# Accuracy: 1279 / 1400 = 0.9136 
# ===== Test Accuracy =====
# Accuracy: 208 / 250 = 0.8320 
# Time for training and test: 52.21 seconds

# lamda = 10
# ===== Train Accuracy =====
# Accuracy: 1276 / 1400 = 0.9114 
# ===== Test Accuracy =====
# Accuracy: 208 / 250 = 0.8320 
# Time for training and test: 46.01 seconds





# you can change the following line to whichever classifier you want to use for
# the bonus.
# i.e to choose NaiveBayes classifier, you can write
# class BonusClassifier(NaiveBayesClassifier):
class BonusClassifier(NaiveBayesClassifier):
    def __init__(self):
        super().__init__()
