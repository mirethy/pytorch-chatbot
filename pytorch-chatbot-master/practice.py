
import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bow(t_sentence, words):
    s_words = [stem(word) for word in t_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for index, w in enumerate(words):
        if w in s_words:
            bag[index] = 1
    
    return bag