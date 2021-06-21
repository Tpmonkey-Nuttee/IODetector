import discord
import time
import random

import pickle
import configparser

import pythainlp
import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer

import streamlit as st

_config = configparser.ConfigParser()
_token = _config["SECRET"]["BotToken"]
REPLY_MESSAGES = ["You FOOLS!"]

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
        msg = re.sub(r'#', '', msg)
        # Remove twitter tag like @Nuttee
        msg = re.sub("/(^|[^@\w])@(\w{1,15})\b/", "", msg)
        msg = remove_emoji(msg)

        # Remove punc and English alphabet
        return msg.translate(str.maketrans('', '', string.printable))
  
    def predict(self, text: str):
        text = self.__clean(text)
        if text.strip() == "": return False
        
        vectorized = self.vectorizer(text)
        prediction = self.model.predict( vectorized.toarray() )
        
        return prediction[0] == 1

class BotClient(discord.Client):
    def __init__(self, *args, **kwargs) -> None:
        self.io_model = Model()

        super().__init__(*args, **kwargs)

    async def on_ready(self) -> None:
        print("Bot Connected")
        print(f"Login as {self.user.name}")

    async def on_message(self, message: discord.Message) -> None:
        # Process Message if It doesn't come from Bot.
        if not message.author.bot: await self.process_message(message)

    async def process_message(self, message: discord.Message) -> None:
        t = time.time()
        prediction = self.io_model.predict(message.content)
        
        if prediction:
            reply = f"**{message.author}** {random.choice(REPLY_MESSAGES)}"
            try: await message.channel.send(reply)
            except: pass
        print(f"Finished Processing Message | Time Took: {time.time() - t}s | Return Value: {prediction}")

_intents = discord.Intents.none()
_intents.message = True

bc = BotClient(intents = _intents)
bc.run(_token)
