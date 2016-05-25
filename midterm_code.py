import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.spatial
import os
import csv
import random
import pandas as pd
import math
import operator
from sklearn.datasets import load_iris
from sklearn import cross_validation
from sklearn.metrics import classification_report, accuracy_score
from operator import itemgetter
from collections import Counter
from sklearn import cross_validation

 
np.random.seed( 2503865 ) 
# Load in the dataset

# Data from https://archive.ics.uci.edu/ml/datasets/Semeion+Handwritten+Digit

datafilename = 'semeion.data'


# Some constants
numFeatures = 256
featureNames = [ 'px' + str(idx) for idx in range( numFeatures ) ]
numOutputs = 10
numDigits = numOutputs
# Dataset
inputs = []# 2D array of inputs ( # rows ) x ( # of features )
outputs = []# 2D array of outputs ( # rows ) x ( # of digits )
outputLabels = []# 1D array of length ( # rows ), will contain the actual digit values

# Extra data
unknownCharacters = []# a 2D array like inputs, but for instances where the labels are unknown

# Open and read the dataset Semeion
with open( datafilename ) as f:
    lines = f.readlines()

    for line in lines:
        els = line[:-2].split(' ')
        inputs += [ [ float(s) for s in els[:numFeatures] ] ]
        outputs += [ [ float(s) for s in els[numFeatures:] ] ]
        outputLabels += [ ''.join( els[numFeatures:] ).find('1') ]
    '''
    g=open('inputs.txt','w')
    g.write(str(inputs))
    g.close()
    g=open('labels.txt','w')
    g.write(str(outputLabels))
    g.close()
    '''
# Open and read the unknownCharacters data
with open( 'unknownCharacters.txt' ) as f:
    lines = f.readlines()
    length=len(lines)
    i=0
    for line in lines:
        i+=1
        if i==length:
            els = line[:-2].split(' ')
        else:
            els = line[:-3].split(' ')
        unknownCharacters += [ [ float(s) for s in els[:numFeatures] ] ]    

# Show some examples of the data
print( 'An input looks like: ' )
print( inputs[0] )
print( 'An output looks like: ' )
print( outputs[0] )
# Function for displaying characters
# In: list of length 256
# Out: String representation of the 2D character
# Side effect: Prints the character
def drawCharacter( c ):
    h = 16
    w = 16
    output = ''
    for y in range(h):
        for x in range(w):
            if c[x+y*w] == 0:
                output += '.'
            else:
                output += 'X'
        output += '\n'
    print( output )
    return output
# Display an input from the dataset
drawCharacter(inputs[0])
#Display each of the unknown row
print 'The first unknown row:\n'
drawCharacter(unknownCharacters[0])
print 'The second unknown row:\n'
drawCharacter(unknownCharacters[1])
print 'The third unknown row:\n'
drawCharacter(unknownCharacters[2])
print 'The fourth unknown row:\n'
drawCharacter(unknownCharacters[3])
print 'The last unknown row:\n'
drawCharacter(unknownCharacters[4])


'''
PROBLEM 1:KNN CLASSIFIER
'''
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=25)
neigh.fit(inputs, outputLabels)
#Predict the digits
one_prediction=neigh.predict(unknownCharacters[0])
two_prediction=neigh.predict(unknownCharacters[1])
three_prediction=neigh.predict(unknownCharacters[2])
four_prediction=neigh.predict(unknownCharacters[3])
five_prediction=neigh.predict(unknownCharacters[4])
#Print the predictions
print 'The first unknown row is predicted as: '+str(one_prediction[0])
print 'The second unknown row is predicted as: '+str(two_prediction[0])
print 'The third unknown row is predicted as: '+str(three_prediction[0])
print 'The fourth unknown row is predicted as: '+str(four_prediction[0])
print 'The fifth unknown row is predicted as: '+str(five_prediction[0])
#Find the confidence with which the KNN model predicts the digit
'''
print neigh.predict_proba(unknownCharacters[0])
print neigh.predict_proba(unknownCharacters[1])
print neigh.predict_proba(unknownCharacters[2])
print neigh.predict_proba(unknownCharacters[3])
print neigh.predict_proba(unknownCharacters[4])
'''
#Find the K neighbors of each unknown digit in the training set
one=neigh.kneighbors(unknownCharacters[0])
two=neigh.kneighbors(unknownCharacters[1])
three=neigh.kneighbors(unknownCharacters[2])
four=neigh.kneighbors(unknownCharacters[3])
five=neigh.kneighbors(unknownCharacters[4])
#Check if the nearest neighbour (i.e. the most similar row is the same digit which is predicted
#The indices are zero based
#Important-If you want one based indices, add one to each answer
if one_prediction[0]==outputLabels[one[1][0][0]]:
    print 'The nearest index to 1st unknown digit is: '+str(one[1][0][0])
if two_prediction[0]==outputLabels[two[1][0][0]]:
    print 'The nearest index to 1st unknown digit is: '+str(two[1][0][0])
if three_prediction[0]==outputLabels[three[1][0][0]]:
    print 'The nearest index to 1st unknown digit is: '+str(three[1][0][0])
if four_prediction[0]==outputLabels[four[1][0][0]]:
    print 'The nearest index to 1st unknown digit is: '+str(four[1][0][0])
if five_prediction[0]==outputLabels[five[1][0][0]]:
    print 'The nearest index to 1st unknown digit is: '+str(five[1][0][0])


'''
PROBLEM 2: DECISION TREES
'''

print '\n##############\nDecision Tree Classifier\n##############\n'
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
clf = DecisionTreeClassifier(criterion="entropy", max_depth=5, splitter='best', random_state=0 )#Initializing the Classifier
clf.fit(inputs, outputLabels)#Training the Decision tree classifier on data set
#Importing required modules for visualizing the tree
from sklearn.externals.six import StringIO  
import pydot  
dot_data = StringIO() 
tree.export_graphviz(clf, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("Tree_visualizer.pdf")
#Find the overall accuracy of the decision tree classifier
print 'The Overall Accuracy of Decision Tree Classifier is '+str(clf.score(inputs,outputLabels)*100)+' %'
#To find the accuracy of each digit prediction
#We will have to do this one by one
#So, what should we do is predict the digit using our decision tree classifier and check the actual label
#BWe can check how many are correct
#We will store the number of each digits in our dataset, Initially all set to 0
each_digit=[0]*10
#Now, we will store, how many rows are predicted correctly, Initially all set to 0
predicted_digit=[0]*10
total=len(inputs)
for i in range(total):
    actual_label=outputLabels[i]
    predicted_label=clf.predict(inputs[i])
    each_digit[actual_label]+=1
    if predicted_label==actual_label:
        predicted_digit[actual_label]+=1
#print each_digit #[161, 162, 159, 159, 161, 159, 161, 158, 155, 158]
#print predicted_digit #[149, 138, 103, 117, 129, 118, 147, 63, 64, 101]
#So, the above data means that total number of 0 digits are 161, and 149 out of
#them were predicted correctly as 0. Similarly for each digit.
#So, to find the accuracy of prediction of each digit, we should just divide the
#predicted digit by each_digit
accuracy=[]
for i in range(10):
    a=(float(predicted_digit[i])/float(each_digit[i]))*100
    accuracy+=[round(a,3)]
    print 'The Accuracy of prediction of '+str(i)+' is '+str(round(a,3))+' %'
#print accuracy #[92.547, 85.185, 64.78, 73.585, 80.124, 74.214, 91.304, 39.873, 41.29, 63.924]
#Predicting the Unknown Characters
print 'The first unknown row is predicted as: '+str(clf.predict(unknownCharacters[0])[0])
print 'The first unknown row is predicted as: '+str(clf.predict(unknownCharacters[1])[0])
print 'The first unknown row is predicted as: '+str(clf.predict(unknownCharacters[2])[0])
print 'The first unknown row is predicted as: '+str(clf.predict(unknownCharacters[3])[0])
print 'The first unknown row is predicted as: '+str(clf.predict(unknownCharacters[4])[0])

'''
PROBLEM 3:FORWARD-SELECTION WRAPPER METHOD
'''

features=[[] for i in range(256)]
for i in inputs:
    for j in range(256):
        features[j].append([i[j]])
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
taken=[]
c=0
t=0
accuracy=[]
while c!=25:
    prev=-1
    pin=-1
    c+=1
    prev_t=0
    for i in range(256):
        if c==1:
            g=features[i]
        else:
            x=len(t)
            g=t
            if i in taken:
                continue
            for j in range(x):
                g[j].append(features[i][j][0])
            #print g
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=0 )#Initializing the Classifier
        clf.fit(g, outputLabels)#Training the Decision tree classifier on data set
        a=clf.score(g,outputLabels)*100
        if prev<a:
            prev=a
            prev_t=g
            pin=i
        #print 'The Overall Accuracy of Decision Tree Classifier is '+str(a)+' %'
    taken.append(pin)
    accuracy.append(prev)
    t=prev_t
    print taken
    print accuracy
'''
PROBLEM 4: COMPARING CLASSIFIERS
'''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
X = inputs
y = outputLabels
#the accuracy for the KNN classifier given each fold(k-fold=10)
neigh = KNeighborsClassifier(n_neighbors=25)
scores_knn = cross_val_score(neigh, X, y, cv=10, scoring='accuracy')
print scores_knn
#the accuracy for the ID3 classifier given each fold(k-fold=10)
clf = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=0 )
scores_ID3 = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
print scores_ID3
#calculate statistical significance 
from scipy.stats.distributions import norm
md = scores_knn.mean() - scores_ID3.mean()
print md
se = np.sqrt(np.var(scores_knn)/len(scores_knn) + np.var(scores_ID3)/len(scores_ID3)) 
print scores_knn.var()
print scores_ID3.var()
print se
t = md/se
print t


'''
PROBLEM 5: KMEANS
'''
from sklearn.metrics.cluster import normalized_mutual_info_score

# Part A ---------------------------------------------------------------------

print('Reading Data Set...')
file = open('semeion.data', 'r')

# Number of samples
m = 1593
# Number of features
n = 256
# Number of centroids
k = 10
# Number of labels
L = 10
# 1 column added for the label and 1 for the centroid
dataset = np.zeros([m, n+2])
labelsCount = np.zeros(L)

# Read data set
for i, line in enumerate(file):
    tokens = [float(f) for f in line.split()]
    dataset[i,:n] = tokens[:n]
    label = tokens[-10:].index(1.0)
    dataset[i,-2] = label
    labelsCount[label] += 1
file.close()
print('Date Set loaded.')

def closest(x, centroids):
    '''
    Find the centroid closest to the sample x
    '''
    dmin = float('inf')
    iclosest = 0
    for i, c in enumerate(centroids):
        dist = np.linalg.norm(x - c)
        if dist < dmin:
            dmin = dist
            iclosest = i
    return iclosest

def kmeans(dataset):
    # Randomly initialize centroids
    centroids = np.random.rand(k, n)
    
    # Start iterating
    print('\nK-Means Iterating...')
    for i in range(25):
        print(i+1)
        # Assign a centroid to each sample
        for s, sample in enumerate(dataset):
            dataset[s,-1] = closest(sample[:n], centroids)
        
        # Update centroids
        for j in range(k):
            centroids[j,:] = np.mean(dataset[dataset[:,-1]==j,:n], axis=0)
    print('Done.\n')
    return dataset, centroids
    
def coincidencesTables(dataset):
    coincs = np.zeros([k,k])
    table = []
    for i in range(k):
        table.append(['Cluster ' + str(i)])
        toCentroid = dataset[dataset[:,-1]==i,:]
        for j in range(k):
            count = len(toCentroid[toCentroid[:,-2]==j,1])
            coincs[i,j] = count
            table[i].append(str(count))
    return coincs, table
    
def orderCentroids(centroids, coincs):
    ordered = np.zeros([k, n])
    orders = []
    for i in range(L):
        digitCoincs = coincs[:,i]
        orders.append(np.argmax(digitCoincs))
    for i in range(k):
        ordered[i] = centroids[orders[i]]
    return ordered, orders
    
def predict(dataset, orderedCents):
    for s, sample in enumerate(dataset):
        dataset[s,-1] = closest(sample[:n], orderedCents)
    return dataset
    
def accuracy(predDataSet):
    '''
    Compute accuracy for a data set predicted by k-Means
    '''
    accuracies = []
    for i in range(10):
        TPdata = predDataSet[predDataSet[:,-1]==i,:]
        TP = len(TPdata[TPdata[:,-2]==i,0])
        TNdata = predDataSet[predDataSet[:,-1]!=i,:]
        TN = len(TNdata[TNdata[:,-2]!=i,0])
        P = labelsCount[i]
        N = np.sum(labelsCount) - P
        accuracies.append((TP + TN)/(P + N))
    return accuracies
    
dataset, centroids = kmeans(dataset)
coincs, table = coincidencesTables(dataset)
centroids, order = orderCentroids(centroids, coincs)
dataset = predict(dataset, centroids)
accuracies = accuracy(dataset)
    
# Make report
report = 'Coincidences for each class label\n\n'
report += '{:^70}'.format('Digits')
report += '{:^5} '.format('\n         ')
for i in range(k):
    report += '{:^5} '.format(i)
report += '\n'
        
for row in table:
    for column in row:
        report += '{:^5} '.format(column)
    report += '\n'
    
report += '\n'
report += '{:^30}'.format('Best Matching Clusters')
report += '\n'
report += '{:^10}{:^10}{:^10}\n'.format('Digit', 'Cluster', 'Accuracy') 
for i in range(L):
    report += '{:^10}{:^10}{:^10.4f}\n'.format(i, order[i], accuracies[i])     
   
file = open('problem5a.txt', 'w')
file.write(report)
file.close()

# Part B ---------------------------------------------------------------------
nexper = 10
nmis = np.zeros(nexper)
bestNMI = float('-inf')
for i in range(nexper):
    dataset, centroids = kmeans(dataset)
    coincs, table = coincidencesTables(dataset)
    centroids, order = orderCentroids(centroids, coincs)
    dataset = predict(dataset, centroids)
    nmi = normalized_mutual_info_score(dataset[:,-2], dataset[:,-1])
    nmis[i] = nmi
    if nmi > bestNMI:
        bestNMI = nmi
        accuracies = accuracy(dataset)
        bestorder = order
    
report = "Normalized mutual information\n"
for i in range(nexper):
    report += "Run " + str(i) + ": " + str(nmis[i]) + "\n"

file = open('problem5b.txt', 'w')
file.write(report)
file.close()

# Part C ---------------------------------------------------------------------
report = '{:^30}'.format('Best Matching Clusters') + '\n'
report += '{:^30}'.format('for Lowest NMI') + '\n'
report += '{:^10}{:^10}{:^10}\n'.format('Digit', 'Cluster', 'Accuracy') 
for i in range(L):
    report += '{:^10}{:^10}{:^10.4f}\n'.format(i, bestorder[i], accuracies[i])   
    
file = open('problem5c.txt', 'w')
file.write(report)
file.close()