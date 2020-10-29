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
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    html_temp = """
    <div style="background-color:tomato;"><p style="color:white;font-size:50px;padding:10px">Algoritmo Knn</p></div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    FILE_TYPES = ["csv", "txt"]
    if st.checkbox("Cargar CSV"):
        file = st.file_uploader("Upload file", type=["csv"])
        if st.button("Clasificar"):
            df = pd.read_csv(file)
            clases = df[['Clase']].values
            keys = list(set(clases.ravel()))
            map = [(i + 1) for i in range(len(keys))]
            df = df.replace(keys,map)
            df = df[['x1','x2','Clase']].values
            for i in range(11):
                x = i + 1
                st.markdown("Clasificacion para k = {}".format(x))
                st.write(funciones.knn_prediction(df),x)
                st.pyplot()
            file.close()
        show_file = st.empty()
        if not file:
            show_file.info("Please upload a file of type: " + ", ".join(["csv"]))
            return
    else:
        if st.checkbox("Cargar TXT"):
            file = st.file_uploader("Upload file", type=["txt"])
            if st.button("Clasificar"):  
                df = pd.read_csv(file, sep=";")
                clases = df[['Clase']].values
                keys = list(set(clases.ravel()))
                map = [(i + 1) for i in range(len(keys))]
                df = df.replace(keys,map)
                df = df[['x1','x2','Clase']].values
                for i in range(9):
                    x = i + 1
                    st.markdown("Clasificacion para k = {}".format(x))
                    st.write(funciones.knn_prediction(df),x)
                    st.pyplot()
                
                file.close()
            show_file = st.empty()
            if not file:
                show_file.info("Please upload a file of type: " + ", ".join(["txt"]))
                return
      
if __name__ == '__main__':
    main()