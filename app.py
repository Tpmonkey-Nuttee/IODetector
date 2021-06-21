from model import Model

import streamlit as st

model = Model()

if __name__ == '__main__':
    input = st.empty()
    txt = input.text_input("Insert text:")
    bt = st.button("Text01")
    st.write(txt)
