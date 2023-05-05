import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def generate_data(size, p1, p2):
    coord, label = make_classification(n_samples=size, n_features=2, n_redundant = 0, n_informative=2,
                                       n_clusters_per_class=1, flip_y = 0, weights = (0.5, 0.5))
    data = []
    for l in zip(label, coord[:, 0], coord[:, 1]):
            if l[0] == 0:
                data.append(np.array([-1,l[1],l[2]]))
            else:
                data.append(np.array([1,l[1],l[2]]))

    noise_data = [list(d) for d in data]
    count1 = size * p2 / 2
    countn1 = size * p1 / 2
    while count1 or countn1:
        for i, j in enumerate(noise_data):
            if countn1 == 0 and count1 == 0:
                break

            noise = random.choice([1,-1])

            if count1 == 0 and noise == 1:
                noise = -1
            elif countn1 == 0 and noise == -1:
                noise = 1
            
            if noise == j[0]:
                noise_data[i][0] = - j[0]
                if noise == 1:
                    count1 -= 1
                elif noise == -1:
                    countn1 -= 1  
    
    return data, noise_data

def split_data(data):
    train, test = train_test_split(data,shuffle=True)
    return train, test

def ploting(data, title):
    # Initialize x and y lists for plotting
    x_list = []
    y_list = []
    color_list = []

    # Loop through dictionary and append values to x and y lists
    for key, value in data.items():
        x_list.append(key[0])
        y_list.append(key[1])
        if value == 1:
            color_list.append('red')
        else:
            color_list.append('blue')

    # Plot the points
    fig, ax = plt.subplots()
    ax.scatter(x_list, y_list, color=color_list)
    ax.set_title(title)
    