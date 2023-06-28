# Implementing PCA in Python

## **Introduction to Implementing PCA**

In this lesson, we will be implementing Principal Component Analysis (PCA) using the Python libraries NumPy and scikit-learn.

The motivation of Principal Component Analysis (PCA) is to find a new set of features that are ordered by the amount of variation (and therefore, information) they contain. We can then select a subset of these PCA features. This leaves us with lower-dimensional data that still retains most of the information contained in the larger dataset.

In this lesson, we will:

- Implement PCA in NumPy step-by-step
- Implement PCA in scikit-learn using only a few lines of code
- Use principal components to train a machine learning model
- Visualize principal components using image data

For the next few exercises, we will use a dataset that describes several types of dry beans separated into seven categories.

We will begin by taking a look at the features that describe different categories of beans.

```python
import pandas as pd
import codecademylib3

# Read the csv data as a DataFrame
df = pd.read_csv('./Dry_Bean.csv')

# Remove null and na values
df.dropna()

# 1. Print the DataFrame head
print(df.head())

# 2. Extract the numerical columns
data_matrix = df.drop(columns='Class',inplace=False)

# Extract the classes
classes = df['Class']

print(data_matrix)
```

## **Implementing PCA in NumPy I**

In this exercise, we will perform PCA using the NumPy method `np.linalg.eig`, which performs eigendecomposition and outputs the eigenvalues and eigenvectors.

The ***eigenvalues*** are related to the relative variation described by each principal component. The ***eigenvectors*** are also known as the principal axes. They tell us how to transform (rotate) our data into new features that capture this variation.

To implement this in Python:

```python
correlation_matrix = data_matrix.corr()
eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix).

```

1. First, we generate a correlation matrix using `.corr()`
2. Next, we use `np.linalg.eig()` to perform eigendecompostition on the correlation matrix. This gives us two outputs — the eigenvalues and eigenvectors.

```python
data_matrix = pd.read_csv('./data_matrix.csv')

# 1. Use the `.corr()` method on `data_matrix` to get the correlation matrix 
correlation_matrix = data_matrix.corr()

## Heatmap code:
red_blue = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(correlation_matrix, vmin = -1, vmax = 1, cmap=red_blue)
plt.show()

# 2. Perform eigendecomposition using `np.linalg.eig` 
eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix) 
# 3. Print out the eigenvectors and eigenvalues
print('eigenvectors: ')
print(eigenvectors)

print('eigenvalues: ')
print(eigenvalues)
```

## **Implementing PCA in NumPy II - Analysis**

After performing PCA, we generally want to know how useful the new features are. One way to visualize this is to create a scree plot, which shows the proportion of information described by each principal component.

The proportion of information explained is equal to the relative size of each eigenvalue:

```python
info_prop = eigenvalues / eigenvalues.sum()
print(info_prop)

```

To create a scree plot, we can then plot these relative proportions:

```python
plt.plot(np.arange(1,len(info_prop)+1),
         info_prop,
         'bo-')
plt.show()

```

![https://static-assets.codecademy.com/skillpaths/feature-engineering/PCA/scree.svg](https://static-assets.codecademy.com/skillpaths/feature-engineering/PCA/scree.svg)

From this plot, we see that the first principal component explains about 50% of the variation in the data, the second explains about 30%, and so on.

Another way to view this is to see how many principal axes it takes to reach around 95% of the total amount of information. Ideally, we’d like to retain as few features as possible while still reaching this threshold.

To do this, we need to calculate the cumulative sum of the `info_prop` vector we created earlier:

```python
cum_info_prop = np.cumsum(info_prop)

```

We can then plot these values using matplotlib:

```python
plt.plot(np.arange(1,len(info_prop)+1),
         cum_info_prop,
         'bo-')
plt.hlines(y=.95, xmin=0, xmax=15)
plt.vlines(x=4, ymin=0, ymax=1)
plt.show()

```

![https://static-assets.codecademy.com/skillpaths/feature-engineering/PCA/cum_plot.svg](https://static-assets.codecademy.com/skillpaths/feature-engineering/PCA/cum_plot.svg)

From this plot, we see that four principal axes account for 95% of the variation in the data.

```python
import numpy as np
import pandas as pd
import codecademylib3
import matplotlib.pyplot as plt

eigenvalues = pd.read_csv('eigenvalues.csv')['eigenvalues'].values

# 1. Find the proportion of information for each eigenvector, which is equal to the eigenvalues divided by the sum of all eigenvalues
info_prop = eigenvalues / eigenvalues.sum() 

## Plot the principal axes vs the information proportions for each principal axis

plt.plot(np.arange(1,len(info_prop)+1),info_prop, 'bo-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Axes')
plt.xticks(np.arange(1,len(info_prop)+1))
plt.ylabel('Percent of Information Explained')
plt.show()
plt.clf()

# 2. Find the cumulative sum of the proportions
cum_info_prop = np.cumsum(info_prop)

## Plot the cumulative proportions array

plt.plot(cum_info_prop, 'bo-', linewidth=2)
plt.hlines(y=.95, xmin=0, xmax=15)
plt.vlines(x=3, ymin=0, ymax=1)
plt.title('Cumulative Information percentages')
plt.xlabel('Principal Axes')
plt.xticks(np.arange(1,len(info_prop)+1))
plt.ylabel('Cumulative Proportion of Variance Explained')
plt.show()
```

## **Implementing PCA using Scikit-Learn**

Another way to perform PCA is using the scikit-learn module `sklearn.decomposition.PCA`.

The steps to perform PCA using this method are:

- Standardize the data matrix. This is done by subtracting the mean and dividing by the standard deviation of each column vector.

```python
mean = data.mean(axis=0)
sttd = data.std(axis=0)
data_standardized = (data - mean) / sttd

```

- Perform eigendecomposition by fitting the standardized data. We can access the eigenvectors using the `components_` attribute and the proportional sizes of the eigenvalues using the `explained_variance_ratio_` attribute.

```python
pca = PCA()
components = pca.fit(data_standardized).components_
components = pd.DataFrame(components).transpose()
components.index =  data_matrix.columns
print(components)

```

![https://content.codecademy.com/articles/principal-component-analysis-intro/components.png](https://content.codecademy.com/articles/principal-component-analysis-intro/components.png)

```python
var_ratio = pca.explained_variance_ratio_
var_ratio = pd.DataFrame(var_ratio).transpose()
print(var_ratio)

```

![https://content.codecademy.com/articles/principal-component-analysis-intro/prop_var_explained.png](https://content.codecademy.com/articles/principal-component-analysis-intro/prop_var_explained.png)

This module has many advantages over the NumPy method, including a number of different solvers to calculate the principal axes. This can greatly improve the quality of the results.

## **Projecting the Data onto the principal Axes**

Once we have performed PCA and obtained the eigenvectors, we can use them to project the data onto the first few principal axes. We can do this by taking the dot product of the data and eigenvectors, or by using the `sklearn.decomposition.PCA` module as follows:

```python
from sklearn.decomposition import PCA

# only keep 3 PCs
pca = PCA(n_components = 3)

# transform the data using the first 3 PCs
data_pcomp = pca.fit_transform(data_standardized)

# transform into a dataframe
data_pcomp = pd.DataFrame(data_pcomp)

# rename columns
data_pcomp.columns = ['PC1', 'PC2', 'PC3']

# print the transformed data
print(data_pcomp.head())

```

![https://content.codecademy.com/articles/principal-component-analysis-intro/princomp_head.png](https://content.codecademy.com/articles/principal-component-analysis-intro/princomp_head.png)

Once we have the transformed data, we can look at a scatter plot of the first two transformed features using seaborn or matplotlib. This allows us to view relationships between multiple features at once in 2D or 3D space. Often, the the first 2-3 principal components result in clustering of the data.

Below, we’ve plotted the first two principal components for a dataset of measurements for three different penguin species:

```python
sns.lmplot(x='PC1', y='PC2', data=data_pcomp, hue='species', fit_reg=False)
plt.show()

```

![https://static-assets.codecademy.com/skillpaths/feature-engineering/PCA/pc1pc2.svg](https://static-assets.codecademy.com/skillpaths/feature-engineering/PCA/pc1pc2.svg)

## **PCA as Features**

So far we have used PCA to find principal axes and project the data onto them. We can use a subset of the projected data for modeling, while retaining most of the information in the original (and higher-dimensional) dataset.

For example, recall in the previous exercise that the first four principal axes already contained 95% of the total amount of variance (or information) in the original data. We can use the first four components to train a model, just like we would on the original 16 features.

Because of the lower dimensionality, we should expect training times to be faster. Furthermore, the principal axes ensure that each new feature has no correlation with any other, which can result in better model performance.

In this checkpoint, we will be using the first four principal components as our training data for a Support Vector Classifier (SVC). We will compare this to a model fit with the entire dataset (16 features) using the average likelihood score. Average likelihood is a model evaluation metric; the higher the average likelihood, the better the fit.

```python
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
 
 
data_matrix_standardized = pd.read_csv('./data_matrix_standardized.csv')
classes = pd.read_csv('./classes.csv')
 
# We will use the classes as y
y = classes.Class.astype('category').cat.codes
 
# Get principal components with 4 features and save as X
pca_1 = PCA(n_components=4) 
X = pca_1.fit_transform(data_matrix_standardized) 
 
# Split the data into 33% testing and the rest training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
 
# Create a Linear Support Vector Classifier
svc_1 = LinearSVC(random_state=0, tol=1e-5)
svc_1.fit(X_train, y_train) 
 
# Generate a score for the testing data
score_1 = svc_1.score(X_test, y_test)
print(f'Score for model with 4 PCA features: {score_1}')
 
# Split the original data intro 33% testing and the rest training
X_train, X_test, y_train, y_test = train_test_split(data_matrix_standardized, y, test_size=0.33, random_state=42)
 
# Create a Linear Support Vector Classifier
svc_2 = LinearSVC(random_state=0)
svc_2.fit(X_train, y_train)
 
# Generate a score for the testing data
score_2 = svc_2.score(X_test, y_test)
print(f'Score for model with original features: {score_2}')
```

## **PCA for Images I**

Another way to show the inner workings of PCA is to use an image dataset. An image can be represented as a row in a data matrix, where each feature corresponds to the intensity of a pixel.

In this and the following exercise, we will be using the [Olivetti Faces](https://scikit-learn.org/stable/datasets/real_world.html?highlight=olivetti+faces) image dataset. We will begin by standardizing the images, and then observing the images of faces themselves.

In the next exercise, we will then transform the original data using PCA and re-plot the images using a subset of the principal components. This will allow us to visualize the mechanism by which PCA retains information in the data while reducing the dimensionality.

```python
import numpy as np
from sklearn import datasets
import codecademylib3
import matplotlib.pyplot as plt
 
 
# Download the data from sklearn's datasets
faces = datasets.fetch_olivetti_faces()['data']
 
# 1. Standardize the images using the mean and standard deviation
faces_mean = faces.mean(axis=0)
faces_std = faces.std(axis=0)
faces_standardized = (faces - faces_mean) / faces_std
 
 
# 2. Find the side length of a square image
n_images, n_features = faces_standardized.shape
side_length = int(np.sqrt(n_features))
print(f'Number of features(pixels) per image: {n_features}')
print(f'Square image side length: {side_length}')
 
 # Create an empty 10x8 plot
fig = plt.figure(figsize=(10, 8))
 
# Observe the first 15 images.
for i in range(15):
 
    # Create subplot, remove x and y ticks, and add title
    ax = fig.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    ax.set_title(f'Image of Face: #{i}')
 
    # Get an image from a row based on the current value of i
    face_image = faces_standardized[i]
 
    # Reshape this image into side_length x side_length 
    face_image_reshaped = face_image.reshape(side_length, side_length)
 
    # Show the image
    ax.imshow(face_image_reshaped, cmap=plt.cm.bone)
plt.show()
```

## **PCA for Images II**

Now that we have cleaned up the data, we can perform PCA to retrieve the eigenvalues and eigenvectors.

This can be useful in understanding how PCA works! We can visualize the eigenvectors by plotting them. They actually have a name: ***eigenfaces***. The eigenfaces are the building blocks for all the other faces in the data.

We can also visualize the dimensionality reduction that occurs when we transform the original data using a smaller number of principal components. In the code editor, we’ve provided you with code to:

- Plot the eigenfaces
- Plot the reconstructed faces using a smaller number of transformed features. To start, we’ve used 400 principal components — only 0.9% of the original number of features (pixels)!

## **Exercise # 9: Review**

In this lesson, we have seen how PCA can be implemented using NumPy and scikit-learn. In particular, we have seen how:

- Implementation: scikit-learn provides a more in-depth set of methods and attributes that extend the number of ways to perform PCA or display the percentage of variance for each principal axis.
- Dimensionality reduction: We visualized the data projected onto the principal axes, known as principal components.
- Image classification: We performed PCA on images of faces to visually understand how principal components still retain nearly all the information in the original dataset.
- Improved algorithmic speed/accuracy: Using principal components as input to a classifier, we observed how we could achieve equal or better results with lower dimensional data. Having lower dimensionality also speeds the training.

![Untitled](Implementing%20PCA%20in%20Python%20c531a16819614e93af71711cad07fb35/Untitled.png)