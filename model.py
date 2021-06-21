# Streamlit failed to run discord.py because of Streamlit run our file outside of Main Thread
import pickle

import pythainlp
from pythainlp.corpus import wordnet
from pythainlp.tokenize import word_tokenize
import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

class Model:
    def __init__(self) -> None:
        self.model = pickle.load(open("gnb.pickle", "rb"))
        self.th_stop = pythainlp.corpus.common.thai_stopwords() 
        
        df = pd.read_csv("dataset.csv")
        features_train, features_test, labels_train, labels_test = train_test_split(df['text'], df['is_spam'], test_size=0.3, random_state = 42)
        
        self.vectorizer = TfidfVectorizer(
            analyzer = 'word',  # default settings
            tokenizer= self.__split, # using our own tokenizer
            token_pattern = None # I don't know what is this, but I will put it anyways.
        )
        self.vectorizer.fit(features_train)
  
    def __split(self, text: str):
        # tokenize text using pythainlp tokenizer
        tokens = word_tokenize(text)

        # remove stop words
        tokens = [i for i in tokens if not i in self.th_stop]

        # Find Stemword in Thai
        tokens_temp = []
        for i in tokens:
            w_syn = wordnet.synsets(i)
            if ( len(w_syn) > 0) and ( len( w_syn[0].lemma_names('tha') ) > 0 ):
                tokens_temp.append( w_syn[0].lemma_names('tha')[0] )
            else:
                tokens_temp.append(i)

        tokens = tokens_temp

        # Delete blank space
        tokens = [i for i in tokens if not ' ' in i]

        return tokens
  
    def __clean(self, text: str):
        # Remove Hashtag
        text = re.sub(r'#', '', text)
        # Remove twitter tag like @Nuttee
        text = re.sub("/(^|[^@\w])@(\w{1,15})\b/", "", text)

        # Remove punc and English alphabet
        return text.translate(str.maketrans('', '', string.printable))
  
    def predict(self, text: str):
        text = self.__clean(text)
        if text.strip() == "": return False
        
        print(text)
        print(self.vectorizer)
        vectorized = self.vectorizer.transform( [text] )
        prediction = self.model.predict( vectorized.toarray() )
        
        return prediction[0] == 1
