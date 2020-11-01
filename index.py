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

    st.markdown("**Se realizaran los siguientes pasos:**\n\n 1- Se cargara el dataset en su formato correspondiente (CSV o TXT)\n\n 2- Se cargara un K de limite superior\n\n 3- Se clasificara y graficara para k de 1 a 10\n\n 4- En funcion al K ingresado se determinara el K optimo\n\n 5- Se mostrara una tabla resultado de la validacion cruzada realizada para determinar el K optimo\n\n 6- Se graficara el K optimo obtenido\n\n\n\n")

    if st.sidebar.checkbox("Cargar CSV"):
        file = st.sidebar.file_uploader("Upload file", type=["csv"])
        num_neighbors = st.sidebar.number_input("Ingrese un nro de K vecinos proximos como maximo superior", min_value=0, format="%i", value=1, step=1)
        if st.sidebar.button("Clasificar"):
            df = pd.read_csv(file)
            clases = df[['Clase']].values
            keys = list(set(clases.ravel()))
            map = [(i + 1) for i in range(len(keys))]
            df = df.replace(keys,map)
            df = df[['x1','x2','Clase']].values
            st.markdown("**Graficamos los K de 1 a 10 primeros**")
            for i in range(9):
                st.markdown("Clasificacion con k = {}.".format(i+1))
                fig = funciones.knn_prediction(df,i+1)
                st.pyplot(fig)
            scores, k_optimo = funciones.best_k(df, 5, int(num_neighbors))
            st.markdown("**Realizamos una validacion cruzada para determinar el K optimo**")
            st.markdown("El numero de k optimos es {}".format(k_optimo))
            columns = ["Fold{}".format(i+1) for i in range(5)]
            columns.append("Accuracy")
            frame = pd.DataFrame(scores,index=["K{}".format(i+1) for i in range(int(num_neighbors))] ,columns=columns)
            st.write(frame)
            st.markdown("Clasificacion con k = {}. (Optimo)".format(k_optimo))
            fig = funciones.knn_prediction(df,k_optimo)
            st.pyplot(fig)    
            file.close() 
        show_file = st.empty()
        if not file:
            show_file.info("Please upload a file of type: " + ", ".join(["csv"]))
            return
    elif st.sidebar.checkbox("Cargar TXT"):
        file = st.sidebar.file_uploader("Upload file", type=["txt"])
        num_neighbors = st.sidebar.number_input("Ingrese un nro de K vecinos proximos como maximo superior", min_value=0, format="%i", value=1, step=1)
        if st.sidebar.button("Clasificar"):
            df = pd.read_csv(file, sep=";")
            clases = df[['Clase']].values
            keys = list(set(clases.ravel()))
            map = [(i + 1) for i in range(len(keys))]
            df = df.replace(keys,map)
            df = df[['x1','x2','Clase']].values
            st.markdown("**Graficamos los K de 1 a 10 primeros**")
            for i in range(9):
                st.markdown("Clasificacion con k = {}.".format(i+1))
                fig = funciones.knn_prediction(df,i+1)
                st.pyplot(fig)
            scores, k_optimo = funciones.best_k(df, 5, int(num_neighbors))
            st.markdown("**Realizamos una validacion cruzada para determinar el K optimo**")
            st.markdown("El numero de k optimos es {}".format(k_optimo))
            columns = ["Fold{}".format(i+1) for i in range(5)]
            columns.append("Accuracy")
            frame = pd.DataFrame(scores,index=["K{}".format(i+1) for i in range(int(num_neighbors))] ,columns=columns)
            st.write(frame)
            st.markdown("Clasificacion con k = {}. (Optimo)".format(k_optimo))
            fig = funciones.knn_prediction(df,k_optimo)
            st.pyplot(fig)    
            file.close() 
        show_file = st.empty()
        if not file:
            show_file.info("Please upload a file of type: " + ", ".join(["txt"]))
            return
      
if __name__ == '__main__':
    main()