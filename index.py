import os
import streamlit as st 

# EDA Pkgs
import pandas as pd 
import numpy as np

# Viz Pkgs
import matplotlib.pyplot as plt 
import matplotlib

import funciones

matplotlib.use('Agg')


def main():
    """ Common ML Dataset Explorer """
    st.title("Common ML Dataset Explorer")
    st.subheader("Datasets For ML Explorer with Streamlit")

    html_temp = """
    <div style="background-color:tomato;"><p style="color:white;font-size:50px;padding:10px">Algoritmo Knn</p></div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    def file_selector(folder_path='./datasets'):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox("Select A file",filenames)
        return os.path.join(folder_path,selected_filename)

    filename = file_selector()
    st.info("Su selección {}".format(filename))

    # Read Data
    df = pd.read_csv(filename)
    # Show Dataset

    if st.button("Ver"):
        st.dataframe(df.head())
        df = df.replace(["C1", "C2"],[1,2])
        df = df[['x1','x2','Clase']].values
        st.write(funciones.knn_prediction(df))
        st.pyplot()
    
if __name__ == '__main__':
    main()