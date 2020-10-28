import numpy as np
import matplotlib.pyplot as plt 

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

def knn_prediction(points, k=5, h=0.25, plot=False):
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap (["hotpink","yellowgreen", "lightskyblue","navajowhite","plum"])
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