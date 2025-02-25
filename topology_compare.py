# -*- coding: utf-8 -*-
"""
topology_compare.py: Compares Persistent Images (PI) and
Persistent Landscapes (PL) for classifying data with topological features.
"""

import gudhi
import numpy as np
import pandas as pd
import ot
import os
import wasserstein
import representations
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import random
import matplotlib.pyplot as plt
import time

print(gudhi.__debug_info__)

def cluster(distances, shape_class, k=6):
    """
    Performs K-medoids clustering on a distance matrix. This is not the standard sklearn kmedoids implementation

    Args:
        distances (numpy.ndarray): Distance matrix between data points.
        shape_class (numpy.ndarray): Array of true class labels for each data point.
        k (int): Number of clusters.

    Returns:
        numpy.ndarray: Array of predicted class labels for each data point.
    """
    m = distances.shape[0]  # Number of points

    # Initialize medoids
    curr_medoids = np.array([-1] * k)
    while not len(np.unique(curr_medoids)) == k: # while the array length is not the same as k
        curr_medoids = np.array([random.randint(0, m - 1) for _ in range(k)])
    old_medoids = np.array([-1] * k)
    new_medoids = np.array([-1] * k)

    # Iterate until medoids converge
    while not (old_medoids == curr_medoids).all():
        clusters = assign_points_to_clusters(curr_medoids, distances)

        for curr_medoid in curr_medoids:
            cluster_points = np.where(clusters == curr_medoid)[0]
            new_medoids[curr_medoids == curr_medoid] = compute_new_medoid(cluster_points, distances)

        old_medoids[:] = curr_medoids[:]
        curr_medoids[:] = new_medoids[:]
    return np.array([shape_class[i] for i in clusters])


def assign_points_to_clusters(medoids, distances):
    """Assigns data points to the closest medoid based on the distance matrix."""
    distances_to_medoids = distances[:, medoids]
    clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
    clusters[medoids] = medoids
    return clusters


def compute_new_medoid(cluster, distances):
    """Finds the best medoid for a cluster by minimizing the total distance."""
    mask = np.ones(distances.shape)
    mask[np.ix_(cluster, cluster)] = 0.  # Mask distances within the cluster
    cluster_distances = np.ma.masked_array(data=distances, mask=mask, fill_value=10e9)
    costs = cluster_distances.sum(axis=1)
    return costs.argmin(axis=0, fill_value=10e9)


def load_data(data_path):
    """Loads data from files in the specified directory."""
    pcloud_list = []
    for filename in os.listdir(data_path):
        if filename.endswith(".txt"):  # Assuming data files are text files
            filepath = os.path.join(data_path, filename)
            df = pd.read_table(filepath, header=None, sep=' ')

            # Extract parameters from filename
            parts = filename[12:-4].split('_') # removes un-needed parts of the files
            noise_level = float([0.05, 0.1][parts[0] == '1']) # change str flag to noise level
            pcloud_num = int(parts[1])
            shape_class = int(parts[2])
            homology_dim = int(parts[3])
            diag = df.values

            pcloud = (shape_class, homology_dim, noise_level, diag)
            pcloud_list.append(pcloud)
    return pcloud_list


# Parameters for PI and PL
b = 1  # Persistence scaling factor
weight_func = lambda y: min(y[1] / b, 1) * (y[1] > 0)  # Weight function for PI

pi_resolution = 20
pi_variance = 0.1

pl_num_landscapes = 5
pl_resolution = 80

# Load data
data_path = './ToyData_PD_TextFiles'
pcloud_list = load_data(data_path)

# Lists to store results
acc_list = []
time_list = []

noise_levels = [0.05, 0.1]
homology_dims = [0, 1]
distance_metrics = ['l1', 'l2', 'chebyshev']

# Main Loop
for noise_level in noise_levels:
    for homology_dim in homology_dims:
        for norm in distance_metrics:
            # Filter data based on current parameters
            selected_data = [
                (i[0], i[3])
                for i in pcloud_list
                if (i[1] == homology_dim and i[2] == noise_level)
            ]
            if not selected_data:
                print(f"No data found for {noise_level=}, {homology_dim=}, {norm=}")
                continue

            shape_classes, selected_diags = zip(*selected_data) # unzip the zipped

            # --- Persistent Image Calculation ---
            pi_start_time = time.time()
            pi = representations.vector_methods.PersistenceImage(
                bandwidth=pi_variance, resolution=[pi_resolution, pi_resolution], weight=weight_func
            )
            pi_vecs = pi.fit_transform(selected_diags)
            pi_distance_matrix = pairwise_distances(pi_vecs, metric=norm)
            pi_accuracies = []

            for _ in range(200):
                pi_labels = cluster(pi_distance_matrix, shape_classes)
                pi_accuracies.append(accuracy_score(shape_classes, pi_labels))

            pi_accuracy = np.mean(pi_accuracies)
            pi_time = time.time() - pi_start_time

            # --- Persistent Landscape Calculation ---
            pl_start_time = time.time()
            pl = representations.vector_methods.Landscape(num_landscapes=pl_num_landscapes, resolution=pl_resolution)
            pl_vecs = pl.fit_transform(selected_diags)
            pl_distance_matrix = pairwise_distances(pl_vecs, metric=norm)
            pl_accuracies = []

            for _ in range(200):
                pl_labels = cluster(pl_distance_matrix, shape_classes)
                pl_accuracies.append(accuracy_score(shape_classes, pl_labels))

            pl_accuracy = np.mean(pl_accuracies)
            pl_time = time.time() - pl_start_time

            # Store Results
            acc_list.append((pi_accuracy, pl_accuracy))
            time_list.append((pi_time, pl_time))

# --- Plotting Results ---
labels = [
    f'H{hdim},{norm},{noise}'
    for noise in ['.05', '0.1']
    for hdim in [0, 1]
    for norm in ['L1', 'L2', 'L∞']
]

pi_accuracies = [round(i[0], 2) for i in acc_list]
pl_accuracies = [round(i[1], 2) for i in acc_list]
#plotting the effect of variance
def fitPI(res,var):
    PI_accuracy=[]

    for neta in [0.05,0.1]:
        for hdim in [0,1]:
            selected_sub=list(zip(*[(i[0],i[3])  for i in pcloud_list if (i[1]==hdim and i[2]==neta)]))
            shape_classes=selected_sub[0]

            selected_diags=selected_sub[1]
            PI=representations.vector_methods.PersistenceImage(bandwidth=var,resolution=[res,res],weight=weight_func)
            PI_vecs=PI.fit_transform(selected_diags)
            PI_distance=pairwise_distances(PI_vecs,metric='l2')
            PI_accs_list=[]
            for i in range(50):
                PI_labels=cluster(PI_distance,shape_classes)
                PI_accs_list.append(accuracy_score(shape_classes,PI_labels))
            PI_acc=np.mean(PI_accs_list)
            PI_accuracy.append(PI_acc)
    return PI_accuracy
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(15, 9))
rects1 = ax.bar(x - width / 2, pi_accuracies, width, label='PI')
rects2 = ax.bar(x + width / 2, pl_accuracies, width, label='PL')

ax.set_ylabel('Accuracy')
ax.set_title('Classification Accuracy of PL and PI')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")  # Rotate labels for readability
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',  # Format label to 2 decimal places
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.show()


# Time plot
pi_times = [i[0] for i in time_list]
pl_times = [i[1] for i in time_list]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(15, 7))
rects1 = ax.bar(x - width / 2, pi_times, width, label='PI')
rects2 = ax.bar(x + width / 2, pl_times, width, label='PL')

ax.set_ylabel('Time (seconds)')
ax.set_title('Classification Time of PL and PI')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.legend()
autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.show()

variance_accuracy=[fitPI(20,v/100) for v in range(1,21)]

plt.plot(variance_accuracy)
plt.xlabel("Variance")
plt.ylabel("Accuracy")
plt.ylim((.6,.9))
plt.title("Effect of Variance")
plt.legend(['noise=.05, H0','noise=.05, H1','noise=.1, H0','noise=.1, H1'])

#plotting the effect of resolution
res_accuracy=[fitPI(n,0.1) for n in range(5,101)]
plt.plot(res_accuracy)
plt.xlabel("Resolution")
plt.ylabel("Accuracy")
plt.ylim((.6,.9))
plt.title("Effect of Resolution")
plt.legend(['noise=.05, H0','noise=.05, H1','noise=.1, H0','noise=.1, H1'])