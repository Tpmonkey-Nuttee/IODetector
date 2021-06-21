from model import Model

import streamlit as st

model = Model()

input = st.empty()
txt = input.text_input("Insert text:")
    
prediction = model.predict(txt)    
out = f"This is a normal Message! ({txt})"

if prediction: out = "IO Detected!"
st.write(txt)
