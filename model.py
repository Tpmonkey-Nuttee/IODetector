# Streamlit failed to run discord.py because of Streamlit run our file outside of Main Thread
import pickle

import pythainlp
import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer

class Model:
    def __init__(self) -> None:
        self.model = pickle.load(open("gnb.pickle", "rb"))
        self.th_stop = pythainlp.corpus.common.thai_stopwords() 
        self.vectorizer = TfidfVectorizer(
            analyzer = 'word',  # default settings
            tokenizer= self.__split, # using our own tokenizer
            token_pattern = None # I don't know what is this, but I will put it anyways.
        )
  
    def __split(self, text: str):
        # tokenize text using pythainlp tokenizer
        tokens = pythainlp.tokenize.word_tokenize(text)

        # remove stop words
        tokens = [i for i in tokens if not i in th_stop]

        # Find Stemword in Thai
        tokens_temp = []
        for i in tokens:
            w_syn = pythainlp.corpus.wordnet.synsets(i)
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
        text = remove_emoji(text)

        # Remove punc and English alphabet
        return text.translate(str.maketrans('', '', string.printable))
  
    def predict(self, text: str):
        text = self.__clean(text)
        if text.strip() == "": return False
        
        vectorized = self.vectorizer(text)
        prediction = self.model.predict( vectorized.toarray() )
        
        return prediction[0] == 1
