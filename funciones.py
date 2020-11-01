import numpy as np
import matplotlib.pyplot as plt 
import random
from random import randrange
import mpld3
from mpld3 import plugins

# calculate the Euclidean distance between two vectors
def euclidean_distance(point, dataset, num_neighbors):
    distances = list()
    for row in dataset:
        distance = 0.0
        for i in range(2):
            distance += (point[i] - row[i])**2
        distance = np.sqrt(distance)
        distances.append([distance, row[-1]])
    distances.sort()
    return np.array(distances[0:num_neighbors])

def predict_classification(neighbors):
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

def knn_predict(predictors, point, k):
    sorted_distances = euclidean_distance(point, predictors,k)
    return predict_classification(sorted_distances)    

def knn_prediction(points, k, h=0.25, plot=False):
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap (["blue", "green", "yellow", "red"])
    observation_colormap = ListedColormap (["red","green","blue","darkorange","purple"])
    x_min, x_max, y_min, y_max = (np.min(points[:,0]) - 0.5, np.max(points[:,0]) + 0.5, np.min(points[:,1]) - 0.5, np.max(points[:,1]) + 0.5)
    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xs,ys)

    prediction_grid = np.zeros(xx.shape, dtype=int)
    for i,x in enumerate(xs):
            for j,y in enumerate(ys):
                p = np.array([x,y])
                prediction_grid[j,i] = knn_predict(points, p, k)

    plt.figure(figsize =(15,15))
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)  
    plt.scatter(points[:,0], points[:,1], c = points[:,2], cmap = observation_colormap, s = 50, edgecolor="black", linewidth=0.3)        
    plt.show()
    return plt

# kNN Algorithm
def k_nearest_neighbors(dataset, test, num_neighbors):
    predictions = list()
    for row in test:
        neighbors = euclidean_distance(row, dataset, num_neighbors)
        output = predict_classification(neighbors)
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

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return [correct , (correct / float(len(actual)) * 100.0)]

# Evaluate an algorithm using a cross validation split
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