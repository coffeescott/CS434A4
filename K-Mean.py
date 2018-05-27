import sys
from numpy import genfromtxt
import numpy as np
import csv
import math
import operator
from itertools import combinations
from functools import reduce

	#Function: main
	#usage: defines variables, loads data and prints out results
def main():
	# Print out info
	print("\nASSIGNMENT 4: PART 1: K-MEAN: Crated by Scott Russell")
	#Name and load file into stack
	file = 'data-1.txt' 
	print("\nLoading " + file)
	data = genfromtxt(file, delimiter=',')

	k_value = input("Enter Value for K for this test: ")
	#first do iteration with k = 2
	SSEs, labels, centers, iterations = kmeans(data, k_value)
	SSE_Data = []
	
	#Print out the Entire SSE Set to the screen
	for i in range(iterations):
		tempVal = 0
		for j, SSE in enumerate(SSEs):
			tempVal += SSE[i]
		SSE_Data.append(tempVal)

	for SSE in SSE_Data:
		print(SSE)	
			
	#Function: Kmeans
	#Usage: home base for calling and creating convergence and calculations.
def kmeans(data,k):

	count = 0
	centers = [data[np.random.randint(0, len(data), size=1)].flatten() for i in range(k)]
	oldCenters = None
	labels = []
	SSEs = [[] for i in range(k)]

	MAX_ITER = 50
	print('Calculating k-means for k={0}'.format(k))
	while True:
		if (count >= MAX_ITER or np.array_equal(oldCenters, centers)):
			break
		oldCenters = centers
		count += 1
		labels = getLabels(data, centers)
		centers = getCenters(data, labels, centers)
		#calcSSE(labels, centers, SSEs, data, k)
		values = [[] for i in range(k)]
		for row, label in enumerate(labels):
			values[label].append(data[row])
		for i, value in enumerate(values):
			SSEs[i].append(np.sum((values[i]-centers[i])**2))
	return SSEs, labels, centers, count
	print('Converged after {0} iteration(s)'.format(count))

	#Function: getLabels
	#assigns data to k
def getLabels(data, centers):
	labels = []

	for i, row in enumerate(data):
		prev = np.inf
		best = 0

		for cKey, center in enumerate(centers):
			cur = np.linalg.norm(row - center)

			if cur < prev:
				best = cKey
				prev = cur
		labels.append(best)

	return labels


	#Function: getCenters
	#usage: updates center values for the k clusters
	#Modified Code from open source here: https://mubaris.com/2017/10/01/kmeans-clustering-in-python/
def getCenters(data, labels, centers):
	sums = []
	newCenters = []
	for i in range(len(centers)):
		if labels.count(i) == 0:
			centers[i] = np.random.randint(0, len(data), size=1)

	for i, center in enumerate(centers):
		sums.append([np.zeros(data.shape[1]).flatten(), 0])

	for i, label in enumerate(labels):
		sums[label][0] = np.add(sums[label][0], data[i])
		sums[label][1] += 1

	for i, row in enumerate(sums):
		newCenters.append(np.divide(row[0], row[1]))

	return newCenters

	
if __name__ == '__main__':
	main()
