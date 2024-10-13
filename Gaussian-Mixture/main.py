import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Load the Olivetti faces dataset
data = fetch_olivetti_faces()
X = data.data
y = data.target

# Split the dataset using train_test_split 
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Apply PCA to reduce dimensionality
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.99)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)


# most suitable covariance type 
covariance_types = ['full', 'tied', 'diag', 'spherical']
lowest_bic = np.infty
best_gmm = None
bic_scores = []

for cov_type in covariance_types:
    for n_components in range(5,40):
        gmm = GaussianMixture(n_components=n_components, covariance_type=cov_type, random_state=42)
        gmm.fit(X_train_pca)
        bic = gmm.bic(X_val_pca)
        bic_scores.append(bic)
        
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm

# Plot the variance
plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('PCA explained variance')
plt.grid(True)
plt.show()

# Plot AIC/BIC for each covariance type and the number of components
bic_scores = np.array(bic_scores).reshape(len(covariance_types), len(range(5,40)))

plt.figure(figsize=(8, 6))
for i, cov_type in enumerate(covariance_types):
    plt.plot(range(5,40), bic_scores[i], label=cov_type)

plt.xlabel('Number of components')
plt.ylabel('BIC score')
plt.title('BIC score for different covariance types and components')
plt.legend()
plt.grid(True)
plt.show()

# hard clustering assignments
y_train_pred = best_gmm.predict(X_train_pca)
y_val_pred = best_gmm.predict(X_val_pca)
y_test_pred = best_gmm.predict(X_test_pca)

print("Hard clustering assignments (first 10):", y_test_pred[:10])

# soft clustering probabilities
soft_clusters = best_gmm.predict_proba(X_test_pca)
print("Soft clustering probabilities for first 10:\n", soft_clusters[:10])

# Generate new faces using GMM sample method
generated_faces, _ = best_gmm.sample(10)  # generate 10 new faces
generated_faces_original_space = pca.inverse_transform(generated_faces)

# Visualize the generated faces
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(generated_faces_original_space[i].reshape(64, 64), cmap='gray')
    ax.axis('off')
plt.suptitle('Generated Faces')
plt.show()

# Modify some images 
def modify_images(images):
    modified_images = []
    for img in images:
        flipped = np.fliplr(img.reshape(64, 64))  # Flip 
        darkened = img * 0.5  # Darken
        rotated = np.rot90(img.reshape(64, 64))  # Rotate 
        modified_images.append(flipped.ravel())
        modified_images.append(darkened)
        modified_images.append(rotated.ravel())
    return np.array(modified_images)

X_test_mod = modify_images(X_test[:5])  # Modify 5 images

# Detect anomalies using score_samples
original_scores = best_gmm.score_samples(X_test_pca[:5])
modified_scores = best_gmm.score_samples(pca.transform(scaler.transform(X_test_mod)))

print("Original scores for normal images:", original_scores)
print("Scores for modified images:", modified_scores)

