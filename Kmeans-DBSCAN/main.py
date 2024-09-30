from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Load the Olivetti faces dataset
faces = fetch_olivetti_faces()
X = faces.data
y = faces.target

# Split training set with stratification so that each face is represented
# Using 70% of data for training so that test and validation have enough data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=92)

# Split the temp set into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=92)

# Initialize the classifier
svm = SVC(kernel='linear')

# Apply k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=92)
scores = cross_val_score(svm, X_train, y_train, cv=kf)

# Train on the full training set and validate on the validation set
svm.fit(X_train, y_train)
val_score = svm.score(X_val, y_val)

print(f"Cross-validation scores: {scores}")
print(f"Validation score: {val_score}")


# Apply K-Means clustering
kmeans = KMeans(n_clusters=40, random_state=92)
kmeans.fit(X_train)

# Evaluate clustering using silhouette score
sil_score = silhouette_score(X_train, kmeans.labels_)
print(f'Silhouette Score: {sil_score}')

# Use K-Means centroids as features
X_train_reduced = kmeans.transform(X_train)
X_val_reduced = kmeans.transform(X_val)

# Train a classifier on the reduced dataset
svm.fit(X_train_reduced, y_train)
val_score_reduced = svm.score(X_val_reduced, y_val)
print(f"Validation score on reduced dataset: {val_score_reduced}")


# Normalize the data
X_scaled = StandardScaler().fit_transform(X_train)

# Apply DBSCAN
dbscan = DBSCAN(eps=55, min_samples=3, metric='euclidean') #Facial image data is continuous, so I used euclidean distance
dbscan_labels = dbscan.fit_predict(X_scaled)

# Evaluate the number of clusters and outliers
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)  # -1 is for noise points
n_noise = list(dbscan_labels).count(-1)


#Plot the results 
def plot_dbscan(dbscan, X, size, show_xlabels=True, show_ylabels=True):
    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True
    anomalies_mask = dbscan.labels_ == -1
    non_core_mask = ~(core_mask | anomalies_mask)
    # Obtain core instances from dbscan.components_
    cores = dbscan.components_
    anomalies = X[anomalies_mask]
    non_cores = X[non_core_mask]
    
    plt.scatter(cores[:, 0], cores[:, 1],
                c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")
    plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20,
                c=dbscan.labels_[core_mask])
    plt.scatter(anomalies[:, 0], anomalies[:, 1],
                c="r", marker="x", s=100)
    plt.scatter(non_cores[:, 0], non_cores[:, 1],
                c=dbscan.labels_[non_core_mask], marker=".")
    if show_xlabels:
        plt.xlabel("$x_1$")
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", rotation=0)
    else:
        plt.tick_params(labelleft=False)
    plt.title(f"eps={dbscan.eps:.2f}, min_samples={dbscan.min_samples}")
    plt.grid()
    plt.gca().set_axisbelow(True)

#Print the cluster labels
print("Labels: ",dbscan.labels_[:10])
#cluster index equal to –1 are anomalies
print("Indices of the core instances: ",dbscan.core_sample_indices_[:10])
plot_dbscan(dbscan, X_scaled, size=100)
plt.show()