from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
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

# Initialize the classifier for KFold cross-validation
# Using an SVM with a linear kernel
svm = SVC(kernel='linear')

# Apply k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=92)
scores = cross_val_score(svm, X_train, y_train, cv=kf)

# Train on the full training set and validate on the validation set
svm.fit(X_train, y_train)
val_score = svm.score(X_val, y_val)

print(f"Cross-validation scores: {scores}")
print(f"Validation score: {val_score}")



#Use Agglomerative Clustering

#Define metric and linkage parameters
params = {'euclidean': 'ward', 'minkowski': 'average', 'cosine': 'average'}

for met, link in params.items():
    silhouette_scores = []
    #Find best silhouette score
    for n in range(5,45):
        faces_clf = AgglomerativeClustering(n_clusters=n,metric=met, linkage=link)
        faces_clf.fit(X_train)
        data_labels = faces_clf.labels_
        silhouette_scores.append(silhouette_score(X_train, data_labels))

    #Scatter plot of silhouette scores
    plt.scatter(range(5,45), silhouette_scores)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('\n\nAgglomerative Clustering ' + met)
    plt.show()
    #Best silhouette score
    best_n = np.argmax(silhouette_scores) + 5
    print("Best number of clusters: ", best_n)

    faces_clf = AgglomerativeClustering(n_clusters=best_n,metric=met, linkage=link)
    faces_clf.fit(X_train)
    data_labels = faces_clf.labels_

    print("Classifier metric: ", faces_clf.metric)
    print("Silhouette Score:" , silhouette_score(X_train, data_labels))

        # Plot the data
    plt.scatter(X_train[:, 0], X_train[:, 1], c=data_labels, cmap='viridis')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Hierarchical Clustering ' + faces_clf.metric)
    plt.show()


    #Initialize the classifier for KFold cross-validation
    # Using an SVM with a linear kernel
    faces_svm = SVC(kernel='linear')
    faces_svm.fit(X_train, data_labels)

    # KFold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=92)
    scores = cross_val_score(faces_svm, X_train, data_labels, cv=kf)
    val_score = faces_svm.score(X_val, y_val)
    print(f"Cross-validation scores: {scores}")
    print(f"Validation score: {val_score}")




