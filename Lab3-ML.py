#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
df=pd.read_excel("embeddingsdatalabel.xlsx")
df


# In[26]:


class_a_data = df[df['Label'] == 0]
class_b_data = df[df['Label'] == 1]
intra_class_var_a = np.var(class_a_data[['embed_1', 'embed_2']], ddof=1)
intra_class_var_b = np.var(class_b_data[['embed_1', 'embed_2']], ddof=1)
mean_class_a = np.mean(class_a_data[['embed_1', 'embed_2']], axis=0)
mean_class_b = np.mean(class_b_data[['embed_1', 'embed_2']], axis=0)
inter_class_distance = np.linalg.norm(mean_class_a - mean_class_b)
print(f'Intraclass spread (variance) for Class A: {intra_class_var_a}')
print(f'Intraclass spread (variance) for Class B: {intra_class_var_b}')
print(f'Interclass distance between Class A and Class B: {inter_class_distance}')


# In[6]:


grouped = df.groupby('Label')

# Calculate the class centroids (mean) for each class
class_centroids = {}
for class_label, group_data in grouped:
    class_mean = group_data[['embed_1', 'embed_2']].mean(axis=0)
    class_centroids[class_label] = class_mean

# Print the class centroids
for class_label, centroid in class_centroids.items():
    print(f'Label {class_label} Centroid: {centroid.values}')


# In[7]:


grouped = df.groupby('Label')

# Calculate the standard deviation for each class
class_standard_deviations = {}
for class_label, group_data in grouped:
    class_std = group_data[['embed_1', 'embed_2']].std(axis=0)
    class_standard_deviations[class_label] = class_std

# Print the standard deviations for each class
for class_label, std_deviation in class_standard_deviations.items():
    print(f'Standard Deviation for Label {class_label}: {std_deviation.values}')


# In[10]:


grouped = df.groupby('Label')

# Calculate the mean vectors (centroids) for each class
class_centroids = {}
for class_label, group_data in grouped:
    class_mean = group_data[['embed_1', 'embed_2']].mean(axis=0)
    class_centroids[class_label] = class_mean

# Calculate the distance between mean vectors of different classes
class_labels = list(class_centroids.keys())
num_classes = len(class_labels)
class_distances = {}

for i in range(num_classes):
    for j in range(i + 1, num_classes):
        class_label1 = class_labels[i]
        class_label2 = class_labels[j]
        distance = np.linalg.norm(class_centroids[class_label1] - class_centroids[class_label2])
        class_distances[(class_label1, class_label2)] = distance

# Print the distances between mean vectors
for (class_label1, class_label2), distance in class_distances.items():
    print(f'Distance between Label {class_label1} and Label {class_label2}: {distance}')


# In[4]:


import numpy as np
import matplotlib.pyplot as plt


feature1_data = df['embed_1']

# Define the number of bins (buckets) for the histogram
num_bins = 5

# Calculate the histogram data (hist_counts) and bin edges (bin_edges)
hist_counts, bin_edges = np.histogram(feature1_data, bins=num_bins)

# Calculate the mean and variance of 'Feature1'
mean_feature1 = np.mean(feature1_data)
variance_feature1 = np.var(feature1_data, ddof=1)  # Use ddof=1 for sample variance

# Plot the histogram
plt.hist(feature1_data, bins=num_bins, edgecolor='black', alpha=0.7)
plt.xlabel('Feature1')
plt.ylabel('Frequency')
plt.title('Histogram of Feature1')
plt.grid(True)

# Show the histogram and statistics
plt.show()

# Print the mean and variance of 'Feature1'
print(f'Mean of Feature1: {mean_feature1}')
print(f'Variance of Feature1: {variance_feature1}')


# In[12]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance


vector1 = np.array([df['embed_1'][0], df['embed_2'][0]])
vector2 = np.array([df['embed_1'][3], df['embed_2'][3]])

# Define a range of values for 'r'
r_values = range(1, 11)

# Calculate Minkowski distances for different 'r' values
distances = [distance.minkowski(vector1, vector2, p=r) for r in r_values]

# Create a plot to observe the nature of the graph
plt.plot(r_values, distances, marker='o', linestyle='-')
plt.xlabel('r')
plt.ylabel('Minkowski Distance')
plt.title('Minkowski Distance vs. r')
plt.grid(True)
plt.show()


# In[5]:


import numpy as np
from sklearn.model_selection import train_test_split

selected_classes = [0, 1]
selected_data = df[df['Label'].isin(selected_classes)]

# Define your features (X) and target (y)
X = selected_data[['embed_1', 'embed_2']]
y = selected_data['Label']

# Split the dataset into a train set (70%) and a test set (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Now, you have your train and test sets for binary classification
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# In[6]:


import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Assuming you have already split your data into X_train and y_train
# If not, please refer to the previous code for splitting the data.

# Create a k-NN classifier with k=3
neigh = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier to your training data
neigh.fit(X_train, y_train)


# In[7]:


accuracy = neigh.score(X_test, y_test)

# Print the accuracy report
print("Accuracy:", accuracy)


# In[85]:


# Assuming you have already trained your k-NN classifier (neigh) and have the X_test data.
# If not, please refer to the previous code for training and splitting the data.

# Let's say you want to predict the class for a specific test vector (test_vect)
# You can replace test_vect with the actual feature values you want to classify.
test_vect = [[0.009625,0.003646 ]]  # Replace with the feature values you want to classify

# Use the predict() function to classify the test vector
predicted_class = neigh.predict(test_vect)

# Print the predicted class
print("Predicted Class:", predicted_class[0])


# In[86]:


import numpy as np

# Assuming you have X_test as a Pandas DataFrame or any other data type
# Convert it to a NumPy array
X_test_array = np.array(X_test)

# Now, you can use X_test_array for predictions
predicted_classes = neigh.predict(X_test_array)

# Print the predicted classes for the entire test set
print("Predicted Classes for the Test Set:")
print(predicted_classes)


# In[88]:


predicted_classes = neigh.predict(test_vect)

# Print the predicted classes for the specific test vectors
print("Predicted Classes for the Test Vectors:")
print(predicted_classes)


# In[90]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Assuming you have your dataset and it's loaded as X and y
# If not, please load your dataset accordingly.

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the range of k values
k_values = range(1, 12)  # Values from 1 to 11

# Lists to store accuracy scores for k-NN and NN
knn_accuracies = []
nn_accuracies = []

# Iterate over different values of k
for k in k_values:
    # Train k-NN classifier with k
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    
    # Predict using k-NN
    knn_predictions = knn_classifier.predict(X_test)
    
    # Calculate accuracy for k-NN
    knn_accuracy = accuracy_score(y_test, knn_predictions)
    knn_accuracies.append(knn_accuracy)

    # Train NN classifier with k=1
    nn_classifier = KNeighborsClassifier(n_neighbors=1)
    nn_classifier.fit(X_train, y_train)
    
    # Predict using NN
    nn_predictions = nn_classifier.predict(X_test)
    
    # Calculate accuracy for NN
    nn_accuracy = accuracy_score(y_test, nn_predictions)
    nn_accuracies.append(nn_accuracy)

# Create a plot to compare accuracy scores
plt.figure(figsize=(10, 6))
plt.plot(k_values, knn_accuracies, marker='o', label='k-NN (k=3)')
plt.plot(k_values, nn_accuracies, marker='o', label='NN (k=1)')
plt.title('Accuracy Comparison: k-NN vs. NN')
plt.xlabel('k Value')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.show()


# In[15]:


from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Assuming you have already trained your k-NN classifier (neigh) and have X_train and y_train for training data,
# and X_test and y_test for test data.

# Train the k-NN classifier
neigh.fit(X_train, y_train)

# Predictions for training and test data
y_train_pred = neigh.predict(X_train)
y_test_pred = neigh.predict(X_test)

# Confusion matrix for training data
confusion_matrix_train = confusion_matrix(y_train, y_train_pred)

# Confusion matrix for test data
confusion_matrix_test = confusion_matrix(y_test, y_test_pred)

# Precision, recall, and F1-score for training data
precision_train = precision_score(y_train, y_train_pred, average='weighted')
recall_train = recall_score(y_train, y_train_pred, average='weighted')
f1_score_train = f1_score(y_train, y_train_pred, average='weighted')

# Precision, recall, and F1-score for test data
precision_test = precision_score(y_test, y_test_pred, average='weighted')
recall_test = recall_score(y_test, y_test_pred, average='weighted')
f1_score_test = f1_score(y_test, y_test_pred, average='weighted')

# Print confusion matrix and performance metrics
print("Confusion Matrix (Training Data):")
print(confusion_matrix_train)
print("\nConfusion Matrix (Test Data):")
print(confusion_matrix_test)

print("\nPerformance Metrics (Training Data):")
print("Precision:", precision_train)
print("Recall:", recall_train)
print("F1-Score:", f1_score_train)

print("\nPerformance Metrics (Test Data):")
print("Precision:", precision_test)
print("Recall:", recall_test)
print("F1-Score:", f1_score_test)





