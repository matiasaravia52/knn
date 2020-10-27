import streamlit as st
import numpy as np
import pandas as pd

training = pd.read_csv('./dataset01.csv')
st.dataframe(training)