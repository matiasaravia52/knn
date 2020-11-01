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

    st.markdown("**Se realizarán los siguientes pasos:**\n\n 1- Se cargará el dataset en su formato correspondiente (CSV o TXT)\n\n 2- Se cargará un K de límite superior\n\n 3- Se clasificará y graficará para K de 1 a 10\n\n 4- En función al K ingresado se determinará el K óptimo\n\n 5- Se mostrará una tabla, resultado de la validación cruzada realizada, para determinar el K óptimo\n\n 6- Se graficará el K óptimo obtenido\n\n\n\n")

    if st.sidebar.checkbox("Cargar CSV"):
        file = st.sidebar.file_uploader("Cargar dataset", type=["csv"])
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
                fig = funciones.prediccion_knn(df,i+1)
                st.pyplot(fig)
            scores, k_optimo = funciones.best_k(df, 5, int(num_neighbors))
            st.markdown("**Realizamos una validacion cruzada para determinar el K optimo**")
            st.markdown("El numero de k optimos es {}".format(k_optimo))
            columns = ["Fold{}".format(i+1) for i in range(5)]
            columns.append("Accuracy")
            frame = pd.DataFrame(scores,index=["K{}".format(i+1) for i in range(int(num_neighbors))] ,columns=columns)
            st.write(frame)
            st.markdown("Clasificacion con k = {}. (Optimo)".format(k_optimo))
            fig = funciones.prediccion_knn(df,k_optimo)
            st.pyplot(fig)    
            file.close() 
        show_file = st.empty()
        if not file:
            show_file.info("Cargue un dataset con formato: " + ", ".join(["csv"]))
            return
    elif st.sidebar.checkbox("Cargar TXT"):
        file = st.sidebar.file_uploader("Cargar dataset", type=["txt"])
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
                fig = funciones.prediccion_knn(df,i+1)
                st.pyplot(fig)
            scores, k_optimo = funciones.best_k(df, 5, int(num_neighbors))
            st.markdown("**Realizamos una validacion cruzada para determinar el K optimo**")
            st.markdown("El numero de k optimos es {}".format(k_optimo))
            columns = ["Fold{}".format(i+1) for i in range(5)]
            columns.append("Accuracy")
            frame = pd.DataFrame(scores,index=["K{}".format(i+1) for i in range(int(num_neighbors))] ,columns=columns)
            st.write(frame)
            st.markdown("Clasificacion con k = {}. (Optimo)".format(k_optimo))
            fig = funciones.prediccion_knn(df,k_optimo)
            st.pyplot(fig)    
            file.close() 
        show_file = st.empty()
        if not file:
            show_file.info("Cargue un dataset con formato: " + ", ".join(["txt"]))
            return
      
if __name__ == '__main__':
    main()