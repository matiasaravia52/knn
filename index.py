import os
import streamlit as st 
import streamlit.components.v1 as stc
from typing import Dict

import io
from io import StringIO

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

def format_dataset(df):
    clases = df[['Clase']].values
    keys = list(set(clases.ravel()))
    map = [(i + 1) for i in range(len(keys))]
    df = df.replace(keys,map)
    df = df[['x1','x2','Clase']].values
    return df

def clasificar(df, num_neighbors):
    st.markdown("**Graficamos los K de 1 a 10 primeros**")
    funciones.prediccion_knn(df,num_neighbors)  

def calc_optimo(df, num_neighbors):
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

def load_csv(data_file, num_neighbors, checked_stocks):
    df = pd.read_csv(data_file)
    df = format_dataset(df)
    for stock in checked_stocks:
        if stock == "Clasificar":
            clasificar(df, num_neighbors)
        elif stock == "Calcular el K optimo":
            calc_optimo(df, num_neighbors)

def load_txt(data_file, num_neighbors, checked_stocks):
    df = pd.read_csv(data_file, sep=";")
    df = format_dataset(df)
    for stock in checked_stocks:
        if stock == "Clasificar":
            clasificar(df, num_neighbors)
        elif stock == "Calcular el K optimo":
            calc_optimo(df, num_neighbors)            
def main():  
    html_temp = """
    <div style="background-color:tomato;"><p style="color:white;font-size:50px;padding:10px">Algoritmo Knn</p></div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    if st.sidebar.checkbox("Cargar CSV"):
        num_neighbors = st.number_input("Ingrese un nro de K vecinos proximos como maximo superior", min_value=0, format="%i", value=1, step=1)
        data_file = st.file_uploader("Cargar dataset", type=["csv"])
        st.markdown("**Seleccione lo que desea realizar: **")
        stocks = ["Clasificar", "Calcular el K optimo"]
        check_boxes = [st.checkbox(stock, key=stock) for stock in stocks]
        checked_stocks = [stock for stock, checked in zip(stocks, check_boxes) if checked]
        if data_file is not None and num_neighbors is not None:
            if st.button("Procesar"):
                load_csv(data_file, num_neighbors, checked_stocks)
        show_file = st.empty()
        if not data_file:
            show_file.info("Cargue un dataset con formato: " + ", ".join(["csv"]))
            return
    elif st.sidebar.checkbox("Cargar TXT"):
        num_neighbors = st.number_input("Ingrese un nro de K vecinos proximos como maximo superior", min_value=0, format="%i", value=1, step=1)
        num_neighbors_static[num_neighbors] = num_neighbors
        data_file = st.file_uploader("Cargar dataset", type=["txt"])
        data_file_static[data_file] = data_file
        st.markdown("**Seleccione lo que desea realizar: **")
        stocks = ["Clasificar", "Calcular el K optimo"]
        check_boxes = [st.checkbox(stock, key=stock) for stock in stocks]
        checked_stocks = [stock for stock, checked in zip(stocks, check_boxes) if checked]
        if data_file is not None and num_neighbors is not None:
            if st.button("Procesar"):
                load_txt(data_file, num_neighbors, checked_stocks)
        show_file = st.empty()
        if not data_file:
            show_file.info("Cargue un dataset con formato: " + ", ".join(["txt"]))
            return
      
if __name__ == '__main__':
    main()