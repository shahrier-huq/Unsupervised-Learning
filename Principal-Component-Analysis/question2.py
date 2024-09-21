from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
import numpy as np


#Generate a swiss roll dataset
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=92)

t_binned = np.digitize(t, bins=np.linspace(np.min(t), np.max(t), num=5))  # binning target because Logistic Regression requires discrete target values
X_train, X_test, t_train, t_test = train_test_split(X, t_binned, test_size=0.2, random_state=92)

#Plot the swiss roll dataset
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=t_train)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Swiss Roll Dataset')
plt.show()


#Apply Kernel PCA
rbf_pca = KernelPCA(n_components=2, kernel='rbf', gamma = 0.04)
rbfX_reduced = rbf_pca.fit_transform(X_train)

linear_pca = KernelPCA(n_components=2, kernel='linear', gamma = 0.04)
linearX_reduced = linear_pca.fit_transform(X_train)

sigmoid_pca = KernelPCA(n_components=2, kernel='sigmoid', gamma = 0.04)
sigmoidX_reduced = sigmoid_pca.fit_transform(X_train)

#Plot the reduced datasets
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

axes[0].scatter(rbfX_reduced[:, 0], rbfX_reduced[:, 1], c=t_train)
axes[0].set_title('RBF Kernel PCA')

axes[1].scatter(linearX_reduced[:, 0], linearX_reduced[:, 1], c=t_train)
axes[1].set_title('Linear Kernel PCA')

axes[2].scatter(sigmoidX_reduced[:, 0], sigmoidX_reduced[:, 1], c=t_train)
axes[2].set_title('Sigmoid Kernel PCA')

plt.show()

#Apply Logistic Regression on kPCA reduced datasets and use GridSearchCV to find the best hyperparameters
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kpca', KernelPCA(n_components=2)),
    ('log_reg', LogisticRegression())
])

param_grid = {
    'kpca__gamma': np.linspace(0.01, 0.1, 10), #simple dataset so small gamma choice
    'kpca__kernel': ['rbf', 'sigmoid', 'linear']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5)

grid_search.fit(X_train, t_train)

#Best parameters
print("Best parameters: ", grid_search.best_params_)

#Plot results
best_pipeline = grid_search.best_estimator_

X_test_reduced = best_pipeline.named_steps['kpca'].transform(X_test)
t_pred = best_pipeline.predict(X_test)

# Plot the results
plt.figure(figsize=(10, 8))
plt.scatter(X_test_reduced[:, 0], X_test_reduced[:, 1], c=t_pred, alpha=0.6)
plt.title('Test Set Predictions using Best Kernel PCA and Logistic Regression')
plt.colorbar(label='Predicted Classes')
plt.show()