import os
import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt 
import random
from random import randrange
import mpld3
from mpld3 import plugins

# devuelve las distancias ordenadas
def distancia(point, dataset):
  distance = 0.0
  for i in range(2):
    distance += (point[i] - dataset[:,i:i+1])**2
  distances = np.concatenate((np.sqrt(distance),dataset[:,2:3]),axis=1)  
  distances = distances[distances[:,0].argsort()]    
  return distances    

# devuelve las clasificaciones segun un conjunto de vecinos y un k_neighbors
def clasificacion(neighbors, k_neighbors):
	output_values = [row[-1] for row in neighbors[0:k_neighbors]]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

# realiza la grafica
def graficar_k_neighbors(xs, ys, xx, yy, grid_distancias, k_nei, puntos):
  from matplotlib.colors import ListedColormap
  background = ListedColormap (["blue", "green", "yellow", "red"])
  observation = ListedColormap (["red","green","blue","darkorange","purple"])
  grid = np.zeros(xx.shape, dtype=int)
  for i,x in enumerate(xs):
    for j,y in enumerate(ys):
      distancias_calculadas = grid_distancias[j,i]
      grid[j,i] = clasificacion(distancias_calculadas, (k_nei + 1))            

  plt.figure(figsize =(15,15))
  plt.pcolormesh(xx, yy, grid, cmap = background, alpha = 0.5)  
  scatter = plt.scatter(puntos[:,0], puntos[:,1], c = puntos[:,2], cmap = observation, s = 50, edgecolor="black", linewidth=0.3)  
  keys = list(set(puntos[:,2].ravel()))
  classes = ["Clase {}".format(i + 1) for i in range(len(keys))]
  plt.legend(handles=scatter.legend_elements()[0], labels=classes)   
  st.markdown("Clasificacion con k = {}.".format(k_nei+1))
  st.pyplot(plt.show())

# funcion principal para graficar
def prediccion_knn(puntos, k_max, step=0.25, plot=False):
  x_min, x_max, y_min, y_max = (np.min(puntos[:,0]) - 0.5, np.max(puntos[:,0]) + 0.5, np.min(puntos[:,1]) - 0.5, np.max(puntos[:,1]) + 0.5)
  xs = np.arange(x_min, x_max, step)
  ys = np.arange(y_min, y_max, step)
  xx, yy = np.meshgrid(xs,ys)

  grid_distancias = np.zeros(xx.shape, dtype=list)
  for i,x in enumerate(xs):
    for j,y in enumerate(ys):
      punto = np.array([x,y])
      grid_distancias[j,i] = distancia(punto,puntos)

  [graficar_k_neighbors(xs, ys, xx, yy,grid_distancias, k_nei, puntos) for k_nei in range(k_max)]

# devuelve la distancia heuclidiana
def distancia_heuclidiana(point, row):
  return [np.sqrt(((point[0] - row[0])**2 + (point[1] - row[1])**2)), row[-1]]

# distancia de un punto dado a cada punto de dataset, y como resultado obtenemos un numero dado de los puntos mas cercanos  
def distancia_validation(point, dataset, num_neighbors):
  distances = [distancia_heuclidiana(point, row) for row in dataset]
  distances.sort()
  return np.array(distances[0:num_neighbors])

# en base al resultado anterior de los puntos mas cercanos al dado, clasifica teniendo en cuenta de que clase es la mayoria
def clasificacion_validation(neighbors):
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

# predise un conjunto de puntos dado 
def k_nearest_neighbors(dataset, test, num_neighbors):
  predictions = list()
  for row in test:
    neighbors = distancia_validation(row, dataset, num_neighbors)
    output = clasificacion_validation(neighbors)
    predictions.append(output)
  return np.array(predictions)

# realiza las particiones de dataset para el cross validation
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for _ in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = random.randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(np.array(fold))
	return dataset_split

# En base a las predicciones y las clases de los puntos a clasificar nos calcula el % de eficiencia en la clasificacion
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return [correct , (correct / float(len(actual)) * 100.0)]

def evaluate_algorithm(dataset, n_folds, num_neighbors):
  iterations_correct = list()
  folds = cross_validation_split(dataset, n_folds)
  scores = list()
  for i in range(len(folds)):
    dataset2 = list()
    for j in range(len(dataset)):
      inicio_intervalo = (i*len(folds[i]))
      fin_intervalo = (inicio_intervalo + len(folds[i]) )
      if j not in range(inicio_intervalo , fin_intervalo): 
        dataset2.append(dataset[j])  
    predicted = k_nearest_neighbors(dataset2, folds[i], num_neighbors)
    actual = [int(row[-1]) for row in folds[i]]
    accuracy = accuracy_metric(actual, predicted)
    scores.append(accuracy[-1])
    iterations_correct.append(accuracy)
  iterations_correct.append(sum(scores)/float(len(scores)))    
  return iterations_correct

def best_k(dataset, n_folds, num_neighbors):
  scores = list()
  data_score = list()
  for i in range(num_neighbors):
    score = evaluate_algorithm(dataset, n_folds, (i+1))
    data_score.append(score)
    scores.append(score[-1])
    max_score = max(set(scores))
  return [data_score ,(scores.index(max_score) + 1)]