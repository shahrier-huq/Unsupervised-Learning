import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.datasets import fetch_openml
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

# Load the MNIST data
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# Split the dataset into training & testing data
X_train, y_train = mnist.data[:60000], mnist.target[:60000]
X_test, y_test = mnist.data[60000:], mnist.target[60000:]

# Display digits
fig, axes = plt.subplots(1, 10, figsize=(10, 2))

for i in range(10):
    digit = X_train[y_train == str(i)][0].reshape(28, 28)  # Reshape to 28x28 pixel image
    axes[i].imshow(digit, cmap='gray')
    axes[i].set_title(f'Digit {i}')
    axes[i].axis('off')  # Hide axes for clarity

plt.tight_layout()
plt.show()


# Create the instance of PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

print("Explained Variance Ratio: ", pca.explained_variance_ratio_)


# Scatter plot for PC1 and PC2 projections on 1D
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Scatter plot for PC1
for i in range(10):
    ax[0].scatter(X_pca[y_train == str(i), 0], [0]*len(X_pca[y_train == str(i), 0]), alpha=0.1, color='green')
ax[0].set_title('Projection on PC1 (1D)')
ax[0].set_xlabel('PC1 Value')

# Scatter plot for PC2
for i in range(10):
    ax[1].scatter(X_pca[y_train == str(i), 1], [0]*len(X_pca[y_train == str(i), 1]), alpha=0.1, color='blue')
ax[1].set_title('Projection on PC2 (1D)')
ax[1].set_xlabel('PC2 Value')

plt.show()


# Incremental PCA
n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
    inc_pca.partial_fit(X_batch)
X_reduced = inc_pca.transform(X_train)

#Display the original and compressed target

X_reconstructed = inc_pca.inverse_transform(X_reduced) #to produce 28x28 image from compressed data

fig, axes = plt.subplots(2, 10, figsize=(15, 4))

for i in range(10):
    digit = X_train[y_train == str(i)][0].reshape(28, 28)  # Reshape to 28x28 pixel image
    axes[0, i].imshow(digit, cmap='gray')
    axes[0, i].set_title(f'Original {i}')
    axes[0, i].axis('off') 

    digit2 = X_reconstructed[y_train == str(i)][0].reshape(28, 28)  # Reshape to 28x28 pixel image
    axes[1, i].imshow(digit2, cmap='gray')
    axes[1, i].set_title(f'Compressed {i}')
    axes[1, i].axis('off')  

plt.show()
