import os
import streamlit as st 

# EDA Pkgs
import pandas as pd 
import numpy as np

# Viz Pkgs
import matplotlib.pyplot as plt 
import matplotlib
import random
from random import randrange

import funciones

matplotlib.use('Agg')
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    html_temp = """
    <div style="background-color:tomato;"><p style="color:white;font-size:50px;padding:10px">Algoritmo Knn</p></div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    FILE_TYPES = ["csv", "txt"]
    if st.sidebar.checkbox("Cargar CSV"):
        num_neighbors = st.number_input("Ingrese un nro de K vecinos proximos", min_value=0.0)
        file = st.file_uploader("Upload file", type=["csv"])
        if st.button("Clasificar"):
            df = pd.read_csv(file)
            clases = df[['Clase']].values
            keys = list(set(clases.ravel()))
            map = [(i + 1) for i in range(len(keys))]
            df = df.replace(keys,map)
            df = df[['x1','x2','Clase']].values
            st.markdown("Graficamos los K de 1 a 10 primeros")
            for i in range(9):
                st.markdown("Clasificacion con k = {}.".format(i+1))
                st.write(funciones.knn_prediction(df,i+1))
                st.pyplot()
            scores, k_optimo = funciones.best_k(df, 5, int(num_neighbors))
            st.markdown("El numero de k optimos es {}".format(k_optimo))
            columns = ["Fold{}".format(i+1) for i in range(5)]
            columns.append("Accuracy")
            frame = pd.DataFrame(scores,index=["K{}".format(i+1) for i in range(int(num_neighbors))] ,columns=columns)
            st.write(frame)
            st.markdown("Clasificacion con k = {}. (Optimo)".format(k_optimo))
            st.write(funciones.knn_prediction(df,k_optimo))
            st.pyplot()
            file.close() 
        show_file = st.empty()
        if not file:
            show_file.info("Please upload a file of type: " + ", ".join(["csv"]))
            return
    elif st.sidebar.checkbox("Cargar TXT"):
        num_neighbors = st.number_input("Ingrese un nro de K vecinos proximos", min_value=0.0)
        file = st.file_uploader("Upload file", type=["txt"])
        if st.button("Clasificar"):  
            df = pd.read_csv(file, sep=";")
            clases = df[['Clase']].values
            keys = list(set(clases.ravel()))
            map = [(i + 1) for i in range(len(keys))]
            df = df.replace(keys,map)
            df = df[['x1','x2','Clase']].values
            st.markdown("Graficamos los K de 1 a 10 primeros")
            for i in range(9):
                st.markdown("Clasificacion con k = {}.".format(i+1))
                st.write(funciones.knn_prediction(df,i+1))
                st.pyplot()
            scores, k_optimo = funciones.best_k(df, 5, int(num_neighbors))
            st.markdown("El numero de k optimos es {}".format(k_optimo))
            columns = ["Fold{}".format(i+1) for i in range(5)]
            columns.append("Accuracy")
            frame = pd.DataFrame(scores,index=["K{}".format(i+1) for i in range(int(num_neighbors))] ,columns=columns)
            st.write(frame)
            st.markdown("Clasificacion con k = {}. (Optimo)".format(k_optimo))
            st.write(funciones.knn_prediction(df,k_optimo))
            st.pyplot()
            file.close() 
        show_file = st.empty()
        if not file:
            show_file.info("Please upload a file of type: " + ", ".join(["csv"]))
            return
      
if __name__ == '__main__':
    main()