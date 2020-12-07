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
from time import time
import math

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
    st.markdown("**Graficamos los primeros {} K**".format(num_neighbors))
    funciones.prediccion_knn(df,num_neighbors)  

def calc_optimo(df, num_neighbors, n_folds):
    scores, k_optimo = funciones.best_k(df, int(n_folds), int(num_neighbors))
    st.markdown("El numero de k optimos es {}".format(k_optimo))
    columns = ["Fold{}".format(i+1) for i in range(int(n_folds))]
    columns.append("Accuracy")
    frame = pd.DataFrame(scores,index=["K{}".format(i+1) for i in range(int(num_neighbors))] ,columns=columns)
    st.write(frame)
    return k_optimo

def graficar_optimo(df,k_optimo):
    st.markdown("Clasificacion con k = {}. (Optimo)".format(k_optimo))
    funciones.graficar_k_optimo(df,k_optimo)  

def load_data(file, num_neighbors, checked_stocks, num_neighbors_graficar, sep, n_folds):
    if sep is not None:
        df = pd.read_csv(file, sep = sep)
    else:
        df = pd.read_csv(file)     
    df = format_dataset(df)
    for stock in checked_stocks:
        if stock == "Ingresar la cantidad de graficos a realizar":
            clasificar(df, num_neighbors_graficar)
        elif stock == "Calcular el K optimo" and "Graficar el K optimo" in checked_stocks:
            k_optimo = calc_optimo(df, num_neighbors, n_folds)  
            graficar_optimo(df,k_optimo)
        elif stock == "Calcular el K optimo" and "Graficar el K optimo" not in checked_stocks:
            k_optimo = calc_optimo(df, num_neighbors, n_folds)

     
def main():  
    html_temp = """
    <div style="background-color:tomato;"><p style="color:white;font-size:50px;padding:10px">Algoritmo Knn</p></div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.markdown("**Cargar el dataset a utilizar **")
    try:
        file = st.file_uploader("Seleccione un dataset")
        if file is not None:
            file.seek(0)
            sep = st.selectbox("Seleccione el separador a utilizar", [None, ";", "Tab"])
            st.markdown("**Seleccione lo que desea realizar: **")
            stocks = ["Ingresar la cantidad de graficos a realizar", "Calcular el K optimo", "Graficar el K optimo"]
            check_boxes = [st.checkbox(stock, key=stock) for stock in stocks]
            checked_stocks = [stock for stock, checked in zip(stocks, check_boxes) if checked]
            if file is not None:
                num_neighbors_graficar = 0
                num_neighbors = 0
                n_folds = 0
                if "Ingresar la cantidad de graficos a realizar" in checked_stocks:
                    num_neighbors_graficar = st.number_input("Ingrese la cantidad de K que desea graficar", min_value=0, format="%i", value=1, step=1)
                if "Calcular el K optimo" in checked_stocks:
                    st.markdown("**Realizaremos una validacion cruzada para determinar el K optimo**")
                    num_neighbors = st.number_input("Ingrese un nro de K vecinos proximos como maximo superior", min_value=0, format="%i", value=1, step=1)
                    n_folds = st.number_input("Ingrese la cantidad de particiones que desea para realizar la validacion", min_value=0, format="%i", value=1, step=1)
                if st.button("Procesar"):
                    tiempo_inicial = time()
                    if "Calcular el K optimo" not in checked_stocks and "Graficar el K optimo" in checked_stocks:
                        show_file = st.empty()
                        show_file.error("Para graficar debe calcular el k optimo")
                        return
                    load_data(file, num_neighbors, checked_stocks, num_neighbors_graficar, sep, n_folds)
                    tiempo_final = time()
                    st.markdown("TIEMPO DE EJECUCION: {} ms".format(math.trunc((tiempo_final - tiempo_inicial)*1000)))
            if not file:
                show_file = st.empty()
                show_file.info("Cargue un dataset con formato: " + ", ".join([".csv o .txt"]))
                return
    except Exception as ex:
        print(ex)
        show_file = st.empty()
        show_file.error("Ocurrio un error, intente cargar un nuevo dataset y verifique el separador a utilizar")
      
if __name__ == '__main__':
    main()