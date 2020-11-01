import numpy as np
import matplotlib.pyplot as plt 
import random
from random import randrange
import mpld3
from mpld3 import plugins

def distancia(point, dataset, num_neighbors):
    distances = list()
    for row in dataset:
        distance = 0.0
        for i in range(2):
            distance += (point[i] - row[i])**2
        distance = np.sqrt(distance)
        distances.append([distance, row[-1]])
    distances.sort()
    return np.array(distances[0:num_neighbors])

def clasificacion(neighbors):
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

def predecir(predictors, point, k):
    distancias = distancia(point, predictors,k)
    return clasificacion(distancias)    

def prediccion_knn(puntos, k, step=0.25, plot=False):
    from matplotlib.colors import ListedColormap
    background = ListedColormap (["blue", "green", "yellow", "red"])
    observation = ListedColormap (["red","green","blue","darkorange","purple"])
    x_min, x_max, y_min, y_max = (np.min(puntos[:,0]) - 0.5, np.max(puntos[:,0]) + 0.5, np.min(puntos[:,1]) - 0.5, np.max(puntos[:,1]) + 0.5)
    xs = np.arange(x_min, x_max, step)
    ys = np.arange(y_min, y_max, step)
    xx, yy = np.meshgrid(xs,ys)

    grid = np.zeros(xx.shape, dtype=int)
    for i,x in enumerate(xs):
            for j,y in enumerate(ys):
                punto = np.array([x,y])
                grid[j,i] = predecir(puntos, punto, k)

    plt.figure(figsize =(15,15))
    plt.pcolormesh(xx, yy, grid, cmap = background, alpha = 0.5)  
    scatter = plt.scatter(puntos[:,0], puntos[:,1], c = puntos[:,2], cmap = observation, s = 50, edgecolor="black", linewidth=0.3)  
    classes = ['Clase 1', 'Clase 2']
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)    
    plt.show()
    return plt

def k_nearest_neighbors(dataset, test, num_neighbors):
    predictions = list()
    for row in test:
        neighbors = distancia(row, dataset, num_neighbors)
        output = clasificacion(neighbors)
        predictions.append(output)
    return np.array(predictions)

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