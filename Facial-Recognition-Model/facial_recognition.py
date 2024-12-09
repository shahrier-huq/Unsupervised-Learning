import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import silhouette_score

# Load the UMIST dataset
data = scipy.io.loadmat('umist_cropped.mat')

facedat = data['facedat'][0]
dirnames = data['dirnames'][0]

#Converting to python list because loadmat type is difficult to work with
facedat_array = []
for i in range(20):
    facedat_array.append(facedat[i])

dirnames_array = ['1a', '1b', '1c', '1d', '1e', '1f', '1g', '1h', '1i', '1j', '1k', '1l', '1m', '1n', '1o', '1p', '1q', '1r', '1s', '1t']

images = []
labels = []

for person_index in range(len(facedat_array)):
    person_images = facedat_array[person_index]  # Get the (112, 92, n) array for this person
    num_images = person_images.shape[2]  # Number of images for this person
    
    for i in range(num_images):
        images.append(person_images[:, :, i])  # Append each image
        labels.append(dirnames_array[person_index])  # Append corresponding label

images = np.array(images)  # Convert to a NumPy array
labels = np.array(labels)  

print(images.shape)

# normalize images
images = images / 255.0  

# Stratified split
train_images, temp_images, train_labels, temp_labels = train_test_split(
    images, labels, test_size=0.3, stratify=labels, random_state=42)
val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)

# Dimensionality reduction using PCA
pca = PCA(n_components=100)  # Reduce to 100 components
train_images_flattened = train_images.reshape(train_images.shape[0], -1)
val_images_flattened = val_images.reshape(val_images.shape[0], -1)
test_images_flattened = test_images.reshape(test_images.shape[0], -1)
train_images_pca = pca.fit_transform(train_images_flattened)
val_images_pca = pca.transform(val_images_flattened)
test_images_pca = pca.transform(test_images_flattened)

# Clustering with K-Means
kmeans = KMeans(n_clusters=np.unique(labels).size, init='k-means++', random_state=42)
kmeans.fit(train_images_pca)

# Evaluate clustering using silhouette score
sil_score = silhouette_score(train_images_pca, kmeans.labels_)
print(f'Silhouette Score: {sil_score}')

# Use the K-Means centroids as the new features
train_images = kmeans.transform(train_images_pca)
val_images = kmeans.transform(val_images_pca)
test_images = kmeans.transform(test_images_pca)

# Encode the labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
val_labels_encoded = label_encoder.transform(val_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Create TensorFlow datasets for training, validation, and test sets
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels_encoded))
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels_encoded))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels_encoded))
print(train_dataset.element_spec)

# Batch the datasets
batch_size = 32
train_dataset = train_dataset.batch(batch_size).shuffle(buffer_size=1000)
val_dataset = val_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(20,)),  # Fully connected layer
    Dropout(0.5),  # Regularization
    Dense(np.unique(labels).size, activation='softmax')  # Output layer for classification
])

model.summary()

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_dataset, 
    validation_data=val_dataset,
    batch_size=32,
    epochs=200,
)

# Evaluate on the test set
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy:.2f}")
