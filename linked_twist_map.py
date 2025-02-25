# -*- coding: utf-8 -*-
"""
linked_twist_map.py: Generates and visualizes data from a linked twist map,
used as input for topological data analysis and classification.
"""

import gudhi

import numpy as np
import matplotlib.pyplot as plt

# Print GUDHI debug information
print(gudhi.__debug_info__)

def linked_map_sample(x0, y0, r, size=1000):
    """
    Generates data points from the linked twist map.

    Args:
        x0 (float): Initial x-coordinate.
        y0 (float): Initial y-coordinate.
        r (float): Parameter of the linked twist map.
        size (int): Number of data points to generate.

    Returns:
        numpy.ndarray: A NumPy array of shape (size, 2) containing the generated data points.
    """
    points = [[x0, y0]]
    x_old = x0
    y_old = y0

    for _ in range(size): # changed to _ as i is unused
        x = (x_old + r * y_old * (1 - y_old)) % 1
        y = (y_old + r * x_old * (1 - x_old)) % 1
        points.append([x, y])
        x_old = x
        y_old = y
    return np.array(points)


# Visualization
plt.figure(figsize=(17, 4))
plt.suptitle('Linked Twist Map Truncated Orbits')

# Generate and plot the data
r_values = [4.1, 4.3, 4.6, 5]
for i, r in enumerate(r_values):
    lp_sample = linked_map_sample(np.random.rand(), np.random.rand(), r, 1000)
    plt.subplot(1, 4, i + 1)
    plt.scatter(lp_sample[:, 0], lp_sample[:, 1])

plt.show()