import numpy as np


def most_voted(order_labels, k):

    labels, counts = np.unique(order_labels, return_counts=True)
    return labels[np.argmax(counts)]


def modified_plurality(order_labels, k):
    intersection = True
    while intersection:
        labels, counts = np.unique(order_labels, return_counts=True)
        intersection = number_intersections(counts)
        order_labels = order_labels[:-1]
    return labels[np.argmax(counts)]


def borda_count(order_labels, k):
    labels, counts = np.unique(order_labels, return_counts=True)
    votes = np.zeros(len(counts))
    for i in range(len(order_labels)):
        index = np.where(labels == order_labels[i])[0][0]
        votes[index] = votes[index] + (k-i)

    return labels[np.argmax(votes)]


def number_intersections(counts):
    max_value_counts = max(counts)
    intersection = 0
    for ci in counts:
        if ci == max_value_counts:
            intersection = intersection + 1
    if intersection > 1:
        bol_intersection = True
    else:
        bol_intersection = False

    return bol_intersection