# Questions:
-   Задача кластеризации, цели кластеризации, типы кластерных структур (1 лекция)
-   Предварительная обработка данных: создание векторного пространства признаков, масштабирование данных (2 лекция)
-   Метрические методы кластеризации, метрики (2 лекция)
-   Кластеризация данных с помощью метода k-means (3 лекция)
-   Метод нечёткой кластеризации c-means (4 лекция)
-   Плотностной алгоритм пространственной кластеризации с присутствием шума (DBSCAN) (4 леция)
-   Иерархическая кластеризация (4 лекция)
-   Уменьшение размерности данных (3 лекция)
-   Метод главных компонент (PCA) (3 лекция
-   Стохастическое вложение соседей с t-распределением (t-SNE) (3 лекция)
-   Самоорганизующиеся карты Кохонена (3 лекция, не интересно)
-   Метод классификации k-ближайших соседей
-   Наивный байесовский классификатор
-   Создание классификатора на основе дерева принятия решений
-   Функции ошибки классификаторов (MSE, RMSE, MAE, MAPE), выбор для конкретной задачи 
-   Метод градиентного спуска (стохастический градиентный спуск) 
-   Метод опорных векторов (SVM)
-   Логистическая регрессия
-   Оценка моделей классификации (accuracy, precision, recall, F-мера)  24.10
-   Линейная регрессия 
-   Регуляризация, лассо- и ридж-регрессии
-   Искусственные нейронные сети(05.12) 
-   Функции активации нейронов, выбор для конкретной задачи
-   Метод обратного распространения ошибки(05.12 - до 22:06)
-   Сверточные нейронные сети: операция свертки, пулинг, каналы, карта признаков
-   Построения и обучения моделей в PyTorch
-   Методы компьютерного зрения, библиотека OpenCV
-   Использование каскадов Хаара для обнаружения лиц(Он же метод Виолы-Джонса)
-   Последовательные данные, обработка временных рядов(13.12)
-   Извлечение статистик из временных рядов данных (13.12)
-   Простое экспоненциальное сглаживание
-   Обучение с подкреплением
-   Марковские модели принятия решения 
-   Метод Монте-Карло
-   Метод Q-learning обучения с подкреплением
-   Метод SARSA обучения с подкреплением

-   Clustering task, clustering goals, types of cluster structures (1 lecture)
-   Data preprocessing: creation of a vector feature space, data scaling (2nd lecture)
-   Metric clustering methods, metrics (2nd lecture)
-   Clustering of data using the k-means method (Lecture 3)
-   The c-means fuzzy clustering method (4th lecture)
-   Density algorithm of spatial clustering with the presence of noise (DBSCAN) (4 lec)
-   Hierarchical clustering (4th lecture)
-   Data dimensionality reduction (Lecture 3)
-   The Principal Component Method (PCA) (3rd lecture
-   Stochastic embedding of neighbors with t-distribution (t-SNE) (Lecture 3)
-   Self-organizing Kohonen maps (3rd lecture, not interesting)
-   K-nearest neighbor classification method
-   Naive Bayesian classifier
-   Creation of a classifier based on a decision tree
-   Classifier error functions (MSE, RMSE, MAE, MAPE), selection for a specific task
-   Gradient descent method (stochastic gradient descent)
-   Support Vector Machine (SVM)
-   Logistic regression
-   Evaluation of classification models (accuracy, precision, recall, F-measure) 24.10
-   Linear regression
-   Regularization, lasso and ridge regressions
-   Artificial neural networks(05.12)
-   Neuron activation functions, selection for a specific task
-   Error back propagation method(05.12 - until 22:06)
-   Convolutional neural networks: convolution operation, pooling, channels, feature map
-   Building and training models in PyTorch
-   Computer vision methods, OpenCV library
-   Using Haar cascades for face detection (aka Viola-Jones method)
-   Sequential data, time series processing (13.12)
-   Extracting statistics from time series of data (13.12)
-   Simple exponential smoothing
-   Reinforcement learning
-   Markov decision-making models
-   The Monte Carlo method
-   Q-learning method of reinforcement learning
-   SARSA Reinforcement Learning method



# Clustering

## Clustering task, clustering goals, types of cluster structures (1 lecture)

Clustering is an unsupervised machine learning approach that aims to divide unlabeled data into groups or clusters, where similar data points are placed in the same cluster. The goal of clustering is to identify data structures within a dataset without relying on pre-existing labels or categories. Clustering can be used to improve supervised learning algorithms by using cluster labels as independent variables .

**Clustering Goals:**
- **Identifying Similarities**: Clustering seeks to find similarities within the data by grouping similar data points together.
- **Market Segmentation**: Clustering can help businesses understand customer preferences by segmenting them into different groups based on their purchasing habits.
- **Anomaly Detection**: Clustering can identify unusual patterns or outliers in the data, which could be indicative of fraud or errors.
- **Recommendation Systems**: Clustering can be used to group items or users based on their preferences, which can then be used to make personalized recommendations.

**Types of Cluster Structures:**
- **Hard Clustering**: Each data point is fully assigned to a cluster, and there is no overlap between clusters.
- **Soft Clustering**: Data points are assigned a probability of belonging to each cluster, allowing for some overlap between clusters.

**Clustering Algorithms:**
- **Centroid Models**: Examples include K-Means, where similarity is determined by the distance to the cluster center.
- **Distribution Models**: These models assume that all data points within a cluster follow the same distribution, like the Gaussian distribution.
- **Density Models**: These models identify clusters based on areas of varying data point density, such as DBSCAN and OPTICS.

**Applications of Clustering:**
- **Recommendation Engines**: Grouping users or items based on preferences.
- **Market Segmentation**: Segmenting customers into different groups based on behavior or demographics.
- **Social Network Analysis**: Identifying communities or groups within social networks.
- **Medical Imaging**: Grouping similar images or identifying regions of interest.
- **Anomaly Detection**: Identifying unusual patterns or outliers that may indicate errors or fraud.

**Key Considerations for Clustering:**
- **Outliers**: Treating outliers in your data is important to ensure that they do not distort the clustering results.
- **Cluster Population**: Ensuring that each cluster has a sufficient population is crucial for the validity of the results.
- **Choice of Algorithm**: The choice of clustering algorithm depends on the nature of the data and the specific problem.





## Data preprocessing: creation of a vector feature space, data scaling (2nd lecture)

Data processing is the method of collecting raw data and translating it into usable information. It is a crucial step in the data lifecycle, which involves several stages :

- **Data Collection**: Gathering raw data from various sources.
- **Data Cleaning**: Removing errors, inconsistencies, and duplicates from the data.
- **Data Transformation**: Converting the data into a suitable format for analysis, which may involve normalization, aggregation, or other operations.
- **Data Storage**: Storing the processed data in a database or data warehouse for easy access and retrieval.
- **Data Analysis**: Using statistical and computational methods to derive insights from the data.


There are different types of data processing, including manual data processing, where all operations are done manually without the use of electronic devices or automation software, and automated data processing, which utilizes software tools and algorithms to handle large volumes of data more efficiently 


**Creation of a Vector Feature Space:**
- **Feature Extraction**: Transform raw data into a set of features that can be used by machine learning algorithms. For text data, this might involve converting words into numerical vectors (e.g., using TF-IDF or word embeddings like Word2Vec).
- **Feature Selection**: Choose the most relevant features that contribute to the predictive power of the model. This can be done using methods like correlation analysis, mutual information, or by using feature selection algorithms.

**Data Scaling:**
- **Standardization**: Scale features to have zero mean and unit variance. This is useful for many machine learning algorithms that assume that all features are centered around zero and have the same variance .
   
  ```python
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  ```

- **Normalization**: Scale features to have unit norm. This is useful for algorithms that use a quadratic form, like the dot product, to quantify similarity. It's also the basis of the Vector Space Model used in text classification and clustering .
   
  ```python
  from sklearn.preprocessing import Normalizer
  normalizer = Normalizer()
  X_normalized = normalizer.fit_transform(X)
  ```

- **Min-Max Scaling**: Scale features to a given range, usually  0 to  1. This is useful when you need the scaled features to have a specific range for some reason, such as when you need to ensure that a feature doesn't dominate others .

  ```python
  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler()
  X_scaled = scaler.fit_transform(X)
  ```

- **Robust Scaling**: Use a similar method to standardization but uses the median and interquartile range instead of the mean and standard deviation. This is useful when the data contains outliers .

  ```python
  from sklearn.preprocessing import RobustScaler
  scaler = RobustScaler()
  X_scaled = scaler.fit_transform(X)
  ```

When choosing a scaling method, consider the characteristics of your data and the requirements of the machine learning algorithms you plan to use. For example, if your data contains many outliers, robust scaling might be a better choice than standardization. If you're using a kernel method that requires unit norm, normalization would be appropriate.





## Metric clustering methods, metrics (2nd lecture)

Metric clustering methods in machine learning involve grouping data points based on certain metrics or distances between data points. These methods are often used to find structures in the data that are not obvious from the raw data alone. Here are some common metrics used in clustering:

- **Rand Index**: Measures the similarity between two data clusterings. It is useful when ground-truth cluster labels are available for comparison .
- **Silhouette Score**: Evaluates how close each sample in one cluster is to the samples in the neighboring clusters. It ranges from -1 to  1, with higher scores indicating better-defined clusters .
- **Davies-Bouldin Index**: Measures the average similarity between each cluster and its most similar one. A lower score indicates better separation between clusters .
- **Calinski-Harabasz Index**: Also known as the score index, it calculates the ratio of between-cluster variation to within-cluster variance. Higher values suggest more distinct groups .
- **Adjusted Rand Index**: Compares the resemblance of genuine class labels to predicted cluster labels. It is useful when ground-truth class labels are available .
- **Mutual Information (MI)**: Measures the agreement between the true class labels and the predicted cluster labels, indicating how well the clustering solution captures the underlying structure in the data .
- **Dunn Index**: Measures the ratio between the distance between the clusters and the distance within the clusters. A high Dunn index indicates that the clusters are well-separated and distinct .
- **Jaccard Coefficient**: Measures the similarity between the clustering results and the ground truth, considering the number of data points in each cluster .

When evaluating clustering results, it's important to analyze these metric scores as they provide quantitative indicators of the quality and performance of clustering algorithms. Higher scores generally indicate better clustering performance. The choice of metric depends on the specific requirements of the clustering task, such as whether ground-truth labels are available and the importance of cluster separation versus cluster purity.

Metric clustering methods refer to clustering algorithms that rely on a notion of distance or similarity between data points to group them into clusters. These methods typically involve defining a distance metric or similarity measure between data points and then partitioning the data into clusters such that points within the same cluster are more similar to each other than to points in other clusters. Here are some commonly used metrics and clustering methods:

Distance Metrics:
1. **Euclidean Distance:** Measures the straight-line distance between two points in Euclidean space. It is defined as:
   \[ d(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2} \]
   
2. **Manhattan Distance (City Block Distance):** Measures the distance between two points along axes at right angles. It is defined as:
   \[ d(\mathbf{p}, \mathbf{q}) = \sum_{i=1}^{n} |q_i - p_i| \]
   
3. **Cosine Similarity:** Measures the cosine of the angle between two vectors in a multidimensional space. It is often used for text data or high-dimensional data where the magnitude of the vectors is not important, only the direction. It is defined as:
   \[ \text{cosine\_similarity}(\mathbf{p}, \mathbf{q}) = \frac{\mathbf{p} \cdot \mathbf{q}}{\|\mathbf{p}\| \|\mathbf{q}\|} \]
   D(p,q) = p.q/(||p||*||q||)

Metrics allow you to reduce the time for calculations and reduce the load on calculations.
A metric is a comparison of two objects nearby.
Metric methods, the metric representation allows us to transfer the concept of proximity and remoteness to objects that we have depicted as points in a multidimensional feature space, and now we want to understand objects that are similar to each other in the proximity or remoteness of the points encoding these objects.





## Clustering of data using the k-means method (Lecture 3)

To cluster data using the K-means method, follow these steps:

1. **Choose the Number of Clusters (K)**: The K-means algorithm requires you to specify the number of clusters beforehand. This can be done using methods like the Elbow Method or the Silhouette Score to determine the optimal number of clusters .

2. **Initialize Centroids**: Randomly assign the initial centroids. The K-means++ technique is often used to initialize centroids because it spreads out the initial centroids, reducing the chances of poor outcomes .

3. **Assign Data Points to Clusters**: Calculate the distance from each data point to each centroid. Assign each data point to the cluster with the nearest centroid .

4. **Update Centroids**: Recalculate the centroid of each cluster by taking the mean of all data points assigned to that cluster .

5. **Iterate**: Repeat steps  3 and  4 until the centroids do not change significantly or a maximum number of iterations is reached. The algorithm has converged when the assignments of data points to clusters no longer change .

6. **Evaluate Clusters**: Check the quality of the clusters by looking at the within-cluster variance and ensuring that the clusters are meaningful and representative of the data .

Key Concepts:
- **Cluster Centroids**: The centroids represent the center of each cluster and are updated in each iteration to minimize the total intra-cluster variance.
- **Inertia**: Also known as within-cluster sum of squares (WCSS), inertia measures the compactness of the clusters. It is calculated as the sum of squared distances between each data point and its assigned centroid. Lower inertia indicates better clustering.

Advantages of K-means Clustering:
- Simple and easy to implement.
- Scalable to large datasets.
- Efficient in terms of computational complexity.
- Works well with spherical clusters.

Keep in mind that K-means has some limitations:
- It is sensitive to the initial placement of centroids, which can lead to different results with different initializations .
- It assumes that clusters are spherical and of similar size, which may not always be the case in real-world data .
- It can be sensitive to the scale of the data and the presence of outliers, so it's important to normalize the features before clustering .

For data with complex cluster shapes or when the number of clusters is not known, other clustering algorithms like Spectral Clustering or DBSCAN might be more suitable .



## The c-means fuzzy clustering method (4th lecture)

The C-means (or Fuzzy C-means) clustering method is a variant of the K-means algorithm that allows for fuzzy membership of data points to clusters. Here are the steps to perform C-means clustering:

1. **Choose the Number of Clusters (C)**: Similar to K-means, you start by choosing the number of clusters you want to create.

2. **Initialize Membership Degrees**: Assign membership degrees (u_ij) randomly to each data point for being in the clusters. These degrees represent the degree to which a data point belongs to a particular cluster.

3. **Compute Centroids**: Calculate the centroid for each cluster, which is the center of the cluster.

4. **Update Membership Degrees**: For each data point, compute its membership degrees for each cluster based on the inverse distance to the cluster centroid. This step is iterative and continues until the algorithm has converged, which means the change in membership degrees between two iterations is less than a given sensitivity threshold (ε) .

μij​=∑k=1K​(dik​dij​​)m−12​1​
µij = 1 / (∑k( (Dij / dik)^(2/(m-1)) ))
where dij is the disstance between the the point i and the cluster j

5. **Iterate**: Repeat steps  3 and  4 until the centroids do not change significantly or a maximum number of iterations is reached .

6. **Evaluate Clusters**: Check the quality of the clusters by looking at the membership degrees and ensuring that the clusters are meaningful and representative of the data.

C-means differs from K-means in that it assigns each data point a probability of belonging to each cluster, rather than assigning it to a single cluster. This allows for more flexible clustering, especially when the data points are not clearly separated into distinct clusters .

Key Concepts:
**Fuzziness Parameter (m)**: Controls the degree of fuzziness in the clustering process. Higher values of mm lead to softer assignments and greater overlap between clusters.
**Membership Function**: Determines the degree to which each data point belongs to each cluster. It ensures that data points close to multiple centroids are assigned partial membership to each cluster.


To implement the Fuzzy C-means algorithm, you can use the `fuzzy-c-means` Python package, which provides an API similar to that of Scikit-learn. Here's an example of how to use it:

```python
import numpy as np
from fcmeans import FCM
from matplotlib import pyplot as plt

n_samples =  5000
X = np.concatenate((
    np.random.normal((-2, -2), size=(n_samples,  2)),
    np.random.normal((2,  2), size=(n_samples,  2))
))

fcm = FCM(n_clusters=2)
fcm.fit(X)

# Outputs
fcm_centers = fcm.centers
fcm_labels = fcm.predict(X)

# Plot result
f, axes = plt.subplots(1,  2, figsize=(11,5))
axes.scatter(X[:,0], X[:,1], alpha=.1)
axes.scatter(X[:,0], X[:,1], c=fcm_labels, alpha=.1)
axes.scatter(fcm_centers[:,0], fcm_centers[:,1], marker="+", s=500, c='w')
plt.show()
```

This code snippet demonstrates how to generate synthetic data, fit a Fuzzy C-means model to it, and then visualize the resulting clusters .


## Density algorithm of spatial clustering with the presence of noise (DBSCAN) (4 lec)


DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm that groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions (whose nearest neighbors are too far away). It is particularly useful for datasets where the number of clusters is not known a priori and the clusters can have arbitrary shapes .

Key Concepts of DBSCAN:
- **Core Points**: A data point is considered a core point if it has at least a specified number of neighboring points (minPts) within a specified radius (epsilon, ε).
- **Border Points**: A data point is considered a border point if it is not a core point but is within the ε-neighborhood of a core point.
- **Noise Points (Outliers)**: Data points that are neither core nor border points are considered noise points or outliers.

Here are the key steps in the DBSCAN algorithm:

1. **Find Neighborhoods**: For each point in the dataset, find all points within a specified distance `ε` (eps).

2. **Identify Core Points**: A point is a core point if it has at least `MinPts` neighbors within the `ε` distance.

3. **Expand Clusters**: For each core point, if it has not been assigned to a cluster, create a new cluster and recursively add all points that are within the `ε` distance and have at least `MinPts` neighbors.

4. **Assign Border Points**: Points that are not core points but are within the `ε` distance of a core point are considered border points and are assigned to the same cluster as the core point.

5. **Label Noise**: Points that are not core or border points are considered noise and are not assigned to any cluster.

6. **Iterate**: Repeat the process for all unvisited points in the dataset.

DBSCAN is parameterized by two values: `ε` (eps), which specifies the maximum distance between two samples for them to be considered as in the same neighborhood, and `MinPts`, the number of samples in a neighborhood for a point to be considered as a core point.

DBSCAN is a density-based clustering algorithm, which means it can discover clusters of arbitrary shape and is also capable of finding arbitrarily shaped clusters, even clusters that are entirely surrounded by, but not connected to, a different cluster .

DBSCAN has several advantages:
- It does not require the user to set the number of clusters a priori.
- It can find arbitrarily shaped clusters.
- It is mostly insensitive to the ordering of the data points in the dataset.
- It can find a cluster completely surrounded by (but not connected to) a different cluster.

However, DBSCAN also has some disadvantages:
- It is not entirely deterministic due to the processing order of data points.
- The quality of DBSCAN depends on the distance measure used.
- It may not perform well on datasets with large differences in densities, as the `MinPts` and `ε` parameters may not be chosen appropriately for all clusters.

DBSCAN is available in various programming languages and libraries, such as scikit-learn in Python, ELKI in Java, and R packages like `dbscan` and `fpc`. These implementations often come with optimizations like k-d tree support for Euclidean distance, which can significantly improve the algorithm's performance on large datasets .



## Hierarchical clustering (4th lecture)

Hierarchical clustering is a method of cluster analysis that seeks to build a hierarchy of clusters. It is a way of grouping data points based on their similarities, and it can be particularly useful when you do not know the number of clusters in advance. Hierarchical clustering can be either agglomerative (bottom-up) or divisive (top-down) .

**Agglomerative Hierarchical Clustering**:
- **Bottom-Up Approach**: Each data point starts in its own cluster.
- **Process**: Pairs of clusters are merged as one moves up the hierarchy, based on a chosen distance measure.
- **Stopping Criterion**: The algorithm stops when all data points are merged into a single cluster.
- **Visualization**: The result is often represented in a dendrogram, which is a tree-like diagram showing the sequences of merges .

**Divisive Hierarchical Clustering**:
- **Top-Down Approach**: All data points start in one cluster.
- **Process**: The cluster is split into smaller clusters recursively as one moves down the hierarchy.
- **Stopping Criterion**: The algorithm stops when each data point is in its own cluster.
- **Visualization**: The result is also represented in a dendrogram, but it is an inverted tree showing the sequences of splits .

Hierarchical clustering has several advantages:
- It can handle non-convex clusters and clusters of different sizes and densities.
- It can handle missing data and noisy data.
- It reveals the hierarchical structure of the data, which can be useful for understanding the relationships among the clusters .

The choice between agglomerative and divisive clustering depends on the specific problem and the nature of the data. Hierarchical clustering can be implemented in various programming languages using libraries such as SciPy in Python, R, and Weka .

Linkage Criteria:

The choice of linkage criterion determines how the similarity between clusters is calculated and influences the shape of the resulting dendrogram (tree-like structure representing the hierarchy of clusters). Common linkage criteria include:

- **Single Linkag**: Defines the similarity between two clusters as the minimum distance between any two points in the two clusters.
- **Complete Linkage**: Defines the similarity between two clusters as the maximum distance between any two points in the two clusters.
- **Average Linkage**: Defines the similarity between two clusters as the average distance between all pairs of points in the two clusters.
- **Centroid Linkage**: Defines the similarity between two clusters as the distance between their centroids.

Dendrogram:
A dendrogram is a visual representation of the hierarchy of clusters produced by hierarchical clustering. It is a tree-like diagram where the branches represent the merging or splitting of clusters, and the height of the branches indicates the distance or dissimilarity at which the clusters were merged or split.

Limitations of Hierarchical Clustering:
- Can be computationally expensive, especially for large datasets.
- May be sensitive to the choice of linkage criterion and distance metric.
- Produces a fixed hierarchy once clustering is complete, which may not always match the true underlying structure of the data.


## Data dimensionality reduction (Lecture 3)

Dimensionality reduction is the process of reducing the number of features in a dataset while retaining as much information as possible. It is a data preprocessing step that can be performed before training a machine learning model. This is important for several reasons:

- **Reduces Complexity**: High-dimensional data can be complex and computationally intensive to process. Reducing dimensions can make the model easier to understand and work with.
- **Improves Performance**: Some algorithms perform poorly with high-dimensional data. Dimensionality reduction can help improve the performance of these algorithms by reducing the computational complexity.
- **Visualization**: High-dimensional data is difficult to visualize. Reducing dimensions can make it easier to visualize the data, which can be useful for understanding patterns and relationships.
- **Removes Redundancy**: High-dimensional data often contains redundant features (features that are highly correlated). Dimensionality reduction can help identify and remove these redundant features.
- **Prevents Overfitting**: High-dimensional data can lead to overfitting, where the model fits the training data too closely and does not generalize well to new data. Reducing dimensions can help prevent overfitting.
- **Data Compression**: Dimensionality reduction can lead to data compression, which reduces the storage space required for the dataset .

There are several techniques for dimensionality reduction, including:

- **Principal Component Analysis (PCA)**: This linear transformation technique identifies the axes in the feature space along which the data varies the most and projects the data onto these axes.
- **Singular Value Decomposition (SVD)**: This factorization technique can be used to perform PCA. It decomposes a matrix into singular vectors and singular values.
- **Linear Discriminant Analysis (LDA)**: This technique is used for classification problems and aims to maximize the separability of two or more classes.
- **t-Distributed Stochastic Neighbor Embedding (t-SNE)**: This non-linear technique is particularly good for visualizing high-dimensional data in two or three dimensions.
- **Autoencoders**: These are neural networks that learn to encode data into a lower-dimensional space and then decode it back to the original space.
- **Uniform Manifold Approximation and Projection (UMAP)**: This is a non-linear dimensionality reduction technique that can be used for visualization and feature extraction.

It's important to note that while dimensionality reduction can be beneficial, it can also lead to loss of information. Therefore, it's crucial to choose the right technique and parameters for your specific dataset and problem



## The Principal Component Method (PCA) (3rd lecture

Principal Component Analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. This transformation is defined in such a way that the first principal component has the largest possible variance (that is, accounts for as much of the variability in the data as possible), and each succeeding component in turn has the highest variance possible under the constraint that it is orthogonal to the preceding components .

Here's a brief overview of how PCA works:

1. **Standardize the Data**: PCA is affected by the scales of the features, so it's common to standardize the data before applying PCA. This is done by subtracting the mean and dividing by the standard deviation for each feature.

2. **Compute the Covariance Matrix**: The covariance matrix captures the correlation between different features in the dataset.

3. **Calculate Eigenvalues and Eigenvectors**: The eigenvectors of the covariance matrix are the principal components, and the eigenvalues are the amount of variance explained by each principal component.
    - a covariance matrix is an nXn matrix where Mij has the cov(i,j) the covariance of the two random variables i and j
    - cov(x,y) = 1/n * sum_n( (x-µx)(y-µy) )
    - to calculate the eigenvalues λ we solve the equation: det(C - λI) = 0, this equation has n solutions
    - to find the eigenvectors we solve the equation (C - λI)v = 0 for each λ that we have

4. **Sort Eigenvectors by Eigenvalues**: The eigenvectors are sorted in descending order of their eigenvalues, so that the first eigenvector has the highest eigenvalue.

5. **Transform the Data**: The original data is transformed by the eigenvectors (principal components) to obtain the principal component scores.

6. **Reduce Dimensionality**: The transformed data can now be reduced to a lower-dimensional space by selecting the top k eigenvectors, where k is the number of dimensions you want to reduce the data to.

PCA is widely used in exploratory data analysis and for making predictive models. It helps in visualizing high-dimensional data, reducing noise, and improving the performance of machine learning algorithms by reducing overfitting. It's also used in genetics for identifying genetic variation and in image processing for image compression 




## Stochastic embedding of neighbors with t-distribution (t-SNE) (Lecture 3)

t-Distributed Stochastic Neighbor Embedding (t-SNE) is a non-linear dimensionality reduction technique that is particularly well-suited for visualizing high-dimensional data in a lower-dimensional space, typically two or three dimensions. It was developed by Laurens van der Maaten and Geoffrey Hinton in  2008 .

Here's how t-SNE works:

1. **Pairwise Similarities**: t-SNE computes the pairwise similarities between data points in the high-dimensional space using a Gaussian kernel. The similarity between two points is determined by the inverse of the Euclidean distance between them.

2. **Mapping to Lower Dimensions**: The algorithm then maps the high-dimensional data points onto a lower-dimensional space (e.g.,  2D or  3D) while preserving the pairwise similarities. This is done by minimizing the divergence between the probability distribution of the original high-dimensional data and the lower-dimensional representation.

3. **Optimization**: The optimization process involves iteratively adjusting the positions of the points in the lower-dimensional space to better match the pairwise similarities from the high-dimensional space. This is typically done using gradient descent.

4. **Iterative Process**: t-SNE is an iterative algorithm, and the quality of the resulting visualization can depend on the initial conditions and the convergence of the algorithm.

t-SNE is particularly useful for visualizing complex datasets because it can reveal clusters and non-linear relationships that might not be apparent in the original high-dimensional space. However, it is not deterministic, meaning that it can produce different results on different runs. This can be an advantage when you want to explore the data from different perspectives, but it can also make it less suitable for scenarios where reproducibility is important .

The algorithm is available in various programming languages and libraries, such as scikit-learn in Python, Rtsne in R, and ELKI, which includes a Barnes-Hut approximation for handling large datasets .

In Python, using scikit-learn, you can perform t-SNE as follows:

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=40, random_state=42)
X_train_tsne = tsne.fit_transform(X_train)

print(tsne.kl_divergence_)
```

This code snippet demonstrates how to fit a t-SNE model to a dataset and transform the data into two dimensions .

Choosing Between PCA and t-SNE:

Use PCA when:
- You want to reduce the dimensionality of the data for computational efficiency.
- You want to capture the global structure and maximize variance in the data.
- Your data has linear correlations between features.
 
Use t-SNE when:
- You want to visualize high-dimensional data in a low-dimensional space.
- You want to preserve the local structure and identify clusters in the data.
- Your data has complex, nonlinear relationships between features.


## Self-organizing Kohonen maps (3rd lecture, not interesting)

Kohonen's Self-organizing Maps (SOMs) are a type of artificial neural network used for clustering and dimensionality reduction of high-dimensional data. SOMs are trained using a competitive learning algorithm, which allows the network to learn a low-dimensional representation of the input space of the training samples, typically in a two-dimensional form .

Here's an overview of how SOMs work:

1. **Initialization**: The SOM is initialized with a grid of nodes, which are the neurons of the network. Each node has a weight vector that represents its position in the input space.

2. **Training**: For each input vector in the training set, the SOM searches for the node that is closest to the input vector in the input space. This node is called the Best Matching Unit (BMU).

3. **Competitive Learning**: The BMU and its neighboring nodes are updated to become more like the input vector. This is done by adjusting the weights of the nodes to move them closer to the input vector. The amount of adjustment is determined by the learning rate and the distance to the input vector.

4. **Iterative Process**: The training process is repeated for all input vectors, and the nodes adjust their weights over time to form a map of the input space.

5. **Topological Preservation**: SOMs are designed to preserve the topological properties of the input space. This means that input vectors that are close to each other in the input space will also be close to each other in the SOM.

6. **Visualization**: The resulting map can be visualized as a two-dimensional grid, where each node represents a cluster of input vectors. This allows for the visualization and analysis of high-dimensional data in a more manageable way .

SOMs are particularly useful for visualizing complex datasets and for finding clusters in the data. They can be used for tasks such as image segmentation, document clustering, and anomaly detection. SOMs have been implemented in various programming languages and libraries, such as MATLAB, Python (with the `somoclu` package), and R (with the `kohonen` package) .

# Classification

## K-nearest neighbor classification method

The K-nearest neighbor (K-NN) classification method is a non-parametric, supervised learning algorithm that classifies new data points based on the majority class label of its nearest neighbors. Here's how the K-NN algorithm works for classification:

1. **Select the Number of Neighbors (K)**: Choose the number of neighbors to consider for classification. This is usually a small integer.

2. **Calculate Distances**: For a new data point, calculate the distance to all points in the training set.

3. **Find Nearest Neighbors**: Identify the K training points that are closest to the new data point.

4. **Vote for the Class**: Assign the new data point to the class that is most common among its K nearest neighbors.

5. **Predict**: The new data point is classified into the majority class of its K nearest neighbors.

The K-NN algorithm is considered a lazy learning method because it does not learn a model from the training data but instead stores the training dataset. When a new data point is presented for classification, it calculates the distance to all points in the training set, selects the K nearest neighbors, and assigns the new data point to the class that is most common among its neighbors .

The choice of K is critical, as a small K value will be sensitive to noise, while a large K value may oversmooth the boundaries. There are various methods to select an optimal K, such as cross-validation, which involves partitioning the dataset into subsets and evaluating the K-NN performance on each subset .

The K-NN algorithm can be used for both classification and regression tasks. In regression, the output is the average of the values of the K nearest neighbors .

In Python, the K-NN algorithm can be implemented using libraries like scikit-learn, which provides a KNeighborsClassifier class for classification and a KNeighborsRegressor class for regression .

Here's an example of how to use the KNeighborsClassifier in scikit-learn:

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a KNN classifier with K=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)
```

In this example, the K-NN classifier is trained on the iris dataset with K set to  3, and then it is used to make predictions on the test set .


## Naive Bayesian classifier

The Naive Bayes classifier is a probabilistic classifier based on Bayes' theorem that assumes that the features in the dataset are independent of each other given the class label. This "naive" assumption allows the algorithm to apply the Bayes' theorem in a straightforward manner, which leads to a simple and fast algorithm that is particularly effective for high-dimensional datasets .

**Bayes' Theorem**:
Bayes' theorem provides a way to calculate conditional probabilities. It states that the probability of a hypothesis (or class label) given the evidence (or features) is proportional to the probability of the evidence given the hypothesis, multiplied by the prior probability of the hypothesis, divided by the probability of the evidence:
P(A|B) = P(B|A).P(B)/P(A) - the probability of A given B is the probability of B given A multiplied by the ratio of probabilities of B and A.


**Naive Assumption**:
The "naive" assumption in Naive Bayes is that the features are conditionally independent given the class label. This means that the presence of one feature does not affect the presence of another feature.


Here's how the Naive Bayes classifier works:

1. **Calculate Class Prior**: The class prior is the probability of each class in the dataset. It is calculated as the proportion of instances of each class.

2. **Calculate Likelihood**: For each feature, calculate the likelihood of the feature given the class. This is done by assuming that the features are independent given the class, which is the "naive" assumption. The likelihood is usually estimated using the frequency of the feature in the instances of each class.

3. **Calculate Posterior Probability**: For a new instance, calculate the posterior probability of each class by multiplying the class prior by the product of the likelihoods of each feature given the class.

4. **Classify**: Assign the new instance to the class with the highest posterior probability.

Naive Bayes classifiers are particularly well-suited for text classification tasks, such as spam detection, where the features are often discrete and can be represented as frequency counts or binary presence/absence indicators .

There are several variants of the Naive Bayes classifier, including:

- **Multinomial Naive Bayes**: Assumes that the features are from multinomial distributions, which is useful for discrete data such as frequency counts.
- **Bernoulli Naive Bayes**: Used with Boolean variables, i.e., variables with two values, such as True and False or  1 and  0 .

The Naive Bayes classifier is known for its simplicity, efficiency, and scalability. It can handle high-dimensional data and is less prone to overfitting than other models. However, it relies on the assumption of feature independence, which may not hold true in real-world datasets .

In Python, the Naive Bayes classifier can be implemented using the scikit-learn library, which provides the `GaussianNB`, `MultinomialNB`, and `BernoulliNB` classes for different types of Naive Bayes classifiers .


The Multinomial Naive Bayes classifier is a variant of the Naive Bayes algorithm specifically designed for text classification tasks where features represent word counts or term frequencies. It's particularly well-suited for document classification, spam filtering, and other tasks involving text data.


## Creation of a classifier based on a decision tree

Decision tree classification is a supervised learning method that is used for both classification and regression tasks. It operates by creating a model that predicts the value of a target variable by learning simple decision rules inferred from the data provided. The model is represented as a tree structure, with each node in the tree representing a feature, each branch representing a decision rule, and each leaf node representing an outcome .

Here's a step-by-step explanation of how decision tree classification works:

1. **Root Node**: The decision tree starts with a root node. This is the first decision point and is made based on the most significant feature or the one that has the highest information gain .

2. **Decision Nodes**: Following the root node, the tree splits into branches that lead to decision nodes. These nodes represent the features of the dataset and the decision rules that are applied to the data .

3. **Branches**: Each branch from a decision node represents a possible outcome based on the feature and the value of the feature. The tree splits into further branches based on these outcomes .

4. **Leaf Nodes**: The final nodes of the tree are the leaf nodes, which represent the possible outcomes or class labels. When a new data point is evaluated, it is directed down the tree based on the decisions at each node, and eventually ends up at a leaf node that predicts the class label for that data point .

5. **Pruning**: To prevent overfitting, the tree may be pruned, which involves removing branches that split on features with low importance. This simplifies the model and can improve its ability to generalize to new data .

6. **Evaluation**: The model's performance is evaluated using cross-validation or by using a separate test dataset. The goal is to find a balance between the tree's complexity and its predictive accuracy .

Decision trees are easy to understand and interpret, making them a popular choice for classification tasks. They can handle both numerical and categorical data and can be used for both binary and multi-class classification problems .

Classification Tree:
- Task: Classification trees are used for classification tasks, where the goal is to predict categorical class labels for instances based on their features.
- Output: Each leaf node of a classification tree represents a class label. When making predictions, an instance is classified into the majority class of the leaf node it reaches.
- Splitting Criteria: Classification trees typically use impurity measures such as Gini impurity or entropy to determine the best feature and threshold for splitting the data at each decision node. The goal is to minimize impurity in the resulting subsets.
- Example: A classification tree might be used to predict whether an email is spam or not based on features such as the presence of certain keywords, sender information, and email length.

Regression Tree:
- Task: Regression trees are used for regression tasks, where the goal is to predict a continuous target variable for instances based on their features.
- Output: Each leaf node of a regression tree represents a predicted continuous value. When making predictions, an instance's target value is predicted as the mean (or median) of the target variable of the instances in the leaf node it reaches.
- Splitting Criteria: Regression trees typically use measures such as variance reduction or mean squared error to determine the best feature and threshold for splitting the data at each decision node. The goal is to minimize variance in the resulting subsets.
- Example: A regression tree might be used to predict the price of a house based on features such as the number of bedrooms, square footage, and location.

How to determine the root node and subsequent nodes order:
1. Choosing the Root Node:
  **Splitting Criteria**: Evaluate each feature in the dataset as a potential candidate for the root node based on a splitting criterion such as information gain (for classification) or variance reduction (for regression).
  **Selecting the Best Feature**: Choose the feature that maximizes the splitting criterion, resulting in the best separation of the data into homogeneous subsets. This feature becomes the root node of the decision tree.

2. Growing the Tree:
  **Recursive Splitting**: Once the root node is determined, recursively apply the splitting process to each subset of data created by the root node. This involves evaluating each feature in the subset and selecting the best feature to split on at each decision node.
  **Splitting Criterion**: At each decision node, choose the feature that maximizes the splitting criterion among the candidate features, thus creating child nodes representing different branches of the decision tree.
  **Stopping Criteria**: Continue splitting the data recursively until a stopping criterion is met, such as reaching a maximum tree depth, having a minimum number of samples in leaf nodes, or achieving a minimum level of impurity.

3. Order of Subsequent Nodes:
  **Based on Information Gain or Variance Reduction**: The order of subsequent nodes is determined based on the feature that maximizes the splitting criterion at each decision node. Features with higher information gain or variance reduction are typically chosen first, leading to more discriminatory and informative splits.
  **Repeat Splitting Process**: Continue the splitting process recursively for each subset of data, selecting the best feature to split on at each decision node until the stopping criteria are met.

A Multiway decision tree has multiple children for each node, a dicition tree is usualy binary.


1. **Gini Impurity:** Measures the probability of incorrectly classifying a randomly chosen element if it were randomly labeled according to the distribution of labels in the node.
    Gini(P) = 1 - ∑ P₁²   for i in C (classes in the node)
    after we have calculated the gini inpurity score for each node we callculate the total gini score for all the leavs, we multiply by n/N where n is the number of samples in the leaf and N 
    is the number of all samples in all the leaves (in the set), and we sum all the products     G = ∑ Gini(P)*n(P)/N  n(P) is the number of samples inside the node P
    we choose the features with lowes gini impurity then we do it recursively for each subset, for continus data we find the lowes impurity for different threasholds where a threashold 
    is an average of two consicutive values in a sorted set (sort the set, get the averages of each two, calculate the gini for each average, take the lowes, compare the lowest with the score of other features)
2. **Entropy:** Measures the average level of "impurity" or "disorder" in a set of labels.
    H(P) = - ∑ P₁ * log₂(1/P₁)  for i in C
   
3. **Variance Reduction:** Measures the decrease in variance after a dataset is split.
Variance Reduction=Var(S)−∑i=1m​∣S∣∣Si​∣​×Var(Si​)

where SS is the parent dataset, SiSi​ are the child datasets after splitting, and mm is the number of splits.

### Splitting Criteria Selection:

- In practice, decision tree algorithms typically use one of these splitting criteria to determine the best split at each decision node based on the attribute that maximizes information gain (for classification) or variance reduction (for regression).

- The choice of splitting criteria depends on the problem type, dataset characteristics, and algorithm implementation.

Understanding and choosing the appropriate splitting criteria is crucial for building accurate and interpretable decision tree models.



## Classifier error functions (MSE, RMSE, MAE, MAPE), selection for a specific task

Classifier error functions are used to quantify the performance of a classification model by measuring the disparity between predicted and actual class labels. However, the error functions you mentioned (MSE, RMSE, MAE, MAPE) are typically used in regression tasks rather than classification tasks. For classification tasks, different metrics are more commonly used, such as accuracy, precision, recall, F1-score, and ROC-AUC. Nonetheless, let me briefly explain these error functions and their applications in regression tasks:

1. **Mean Squared Error (MSE)**: Measures the average squared difference between the predicted and actual values. It penalizes large errors more heavily than small errors.
      MSE = 1/n ∑(y₁-ŷ₁)²

2. **Root Mean Squared Error (RMSE)**: RMSE is the square root of the MSE. It's a popular metric because it's in the same units as the target variable.
      RMSE = √(MSE) 

3. **Mean Absolute Error (MAE)**: Measures the average absolute difference between the predicted and actual values. Unlike MSE, MAE is not as sensitive to outliers.
      MAE = 1/n ∑|y₁-ŷ₁|

4. **Mean Absolute Percentage Error (MAPE)**: Measures the average percentage difference between the predicted and actual values. It provides a relative measure of the error, making it easier to interpret across different scales.
      MAPE = 1/n ∑|(y₁-ŷ₁)/y₁| * 100%


## Gradient descent method (stochastic gradient descent)

**Gradient descent** is an optimization algorithm used to minimize a differentiable loss function J(θ)J(θ), where θθ represents the model parameters (weights).
At each iteration, gradient descent computes the gradient of the loss function with respect to the parameters and updates the parameters in the direction of the negative gradient to minimize the loss.

In **stochastic gradient descent (SGD)**, instead of computing the gradient of the loss function using the entire training dataset, the gradient is computed using a single training example (or a small batch of examples) randomly selected from the dataset.
SGD updates the parameters after processing each training example, leading to faster convergence and making it suitable for large datasets or online learning scenarios.

**Mini-batch gradient descent** is a compromise between batch gradient descent (using the entire dataset) and SGD. It updates the parameters using a small randomly selected subset (mini-batch) of the training dataset.
Mini-batch gradient descent offers a balance between the efficiency of SGD and the stability of batch gradient descent.
Stochastic Gradient Descent (SGD) is an iterative optimization algorithm used for training machine learning models. It is a variant of the Gradient Descent algorithm that updates the model's parameters using a single training example at each iteration, rather than the entire dataset .

Here's how SGD works:

1. **Initialize Parameters**: Start with an initial set of parameters.
    for a functon E(w) we take w₀ randomay or using predefined values
2. **Compute Gradient**: For each training example, compute the gradient of the loss function with respect to the parameters.
    wᵗ⁺¹ = wᵗ - r dE/dwᵗ
3. **Update Parameters**: Update the parameters by subtracting the product of the learning rate and the gradient.

4. **Repeat**: Repeat steps   2 and   3 for a specified number of iterations or until the algorithm converges.

The primary advantage of SGD over standard Gradient Descent is that it can handle large datasets efficiently. Instead of computing the gradient over the entire dataset, which can be computationally expensive, SGD computes the gradient for each training example one at a time. This makes SGD particularly suitable for large-scale machine learning problems .

However, SGD has some drawbacks:
- **Noisy Updates**: Because SGD updates the parameters based on one training example at a time, the updates can be noisy, leading to instability in the learning process.
- **Local Minima**: Like standard Gradient Descent, SGD can get stuck in local minima, especially in high-dimensional spaces.

Despite these drawbacks, SGD is widely used in practice because of its computational efficiency and because it often works well in practice, especially when combined with other techniques like momentum and learning rate schedules .

In Python, the `SGD` class from the scikit-learn library provides an implementation of the Stochastic Gradient Descent algorithm. Here's an example of how to use it:

```python
from sklearn.linear_model import SGDRegressor
from sklearn.datasets import make_regression

# Generate a regression problem
X, y = make_regression(n_samples=100, n_features=20, noise=0.1)

# Create an SGD regressor
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)

# Train the regressor
sgd_reg.fit(X, y)

# Make predictions
y_pred = sgd_reg.predict(X)
```

In this example, the `SGDRegressor` is used to fit a linear regression model to the generated dataset. The `max_iter` parameter specifies the maximum number of passes over the training data, `tol` is the stopping criterion, `penalty` is the regularization parameter (set to `None` for no regularization), and `eta0` is the initial learning rate .




## Support Vector Machine (SVM)

Support Vector Machines (SVMs) are a set of supervised learning methods used for classification, regression, and outlier detection. The main idea behind SVMs is to find a hyperplane that best separates the data points of different classes in a feature space .

Here are some key concepts related to SVMs:

- **Hyperplane**: The decision boundary used to separate the data points of different classes. In linear classifications, it is a linear equation (wx+b =  0) .
- **Support Vectors**: These are the data points that are closest to the hyperplane. They play a critical role in determining the hyperplane and margin .
- **Margin**: The distance between the hyperplane and the closest data points (support vectors). SVMs aim to maximize this margin. There are two types of margins: hard margin and soft margin .
- **Kernel**: A mathematical function used to map the original input data points into high-dimensional feature spaces, allowing for non-linear classification even if the data points are not linearly separable in the original input space. Common kernel functions include linear, polynomial, radial basis function (RBF), and sigmoid .
- **Hard Margin**: A hyperplane that perfectly separates the data points of different categories without any misclassifications .
- **Soft Margin**: Allows for some misclassifications or violations of the margin, which is useful when the data is not perfectly separable or contains outliers. The regularization parameter C balances margin maximization and misclassification fines .

SVMs are powerful because they can handle high-dimensional data and non-linear relationships. They are particularly effective in finding the maximum separating hyperplane between different classes available in the target feature .

In Python, you can use the `scikit-learn` library to work with SVMs. Here's an example of how to use SVM for classification:

```python
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM classifier
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

In this example, the `SVC` class from `scikit-learn` is used to create an SVM classifier with a linear kernel and a regularization parameter C of   1. The classifier is then trained on the iris dataset and used to make predictions on the test set .




## Logistic regression


Logistic Regression is a statistical method used for binary classification problems. It is used to predict the probability of a categorical dependent variable using one or more independent variables. The logistic regression model transforms the linear regression function's continuous value output into a categorical value output using a sigmoid function, which maps any real-valued set of independent variables input into a value between  0 and  1 .

The general form of the logistic regression model is:

```
P(Y=1|X) =  1 / (1 + e^-(b0 + b1*X1 + b2*X2 + ... + bn*Xn))
```

Where:
- `P(Y=1|X)` is the probability that the output is   1 given the input variables `X`.
- `b0` is the bias term, also known as the intercept.
- `b1, b2, ..., bn` are the weights or coefficients for the input variables `X1, X2, ..., Xn`.
- `e` is the base of the natural logarithm.

The output of logistic regression is a probability that the given input point belongs to a certain class. In binary classification, this probability can be used to make a decision by choosing a threshold (often   0.5) above which the instance is classified as class   1 and below which it is classified as class   0 .

To optimize the logistic regression model, we use the log-likelihood loss function. The goal is to minimize the cost function, which measures the difference between the predicted probabilities and the actual labels. This is typically done using optimization algorithms such as gradient descent .

In Python, you can use the `LogisticRegression` class from the `scikit-learn` library to create a logistic regression model. Here's an example:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)
```


this regression tries to pridict wether an input X does belong to a class y = 1 with a given probability, so the prediction is a value in [0;1]
it uses the segmoid function as S(Z) = 1/ (1 + e^(-z)) (= a the prediction), and where Z(X) is the linear regression function 
Z(X) = w₀ + w₁x₁ + .... + e  and e is the loss function given as e(x) = -(y log(S(x)) + (1-y) log(S(x)))  where y is the supposed value from training
and S(X) is the segmid for the function Z(X) linear regression
the cost funtion is the average of all losses  J = 1/n ∑e

in order to find the best coefficient w that minimize the cost we can use the gradient descent method
 d/dw J = d/dS J * d/dZ S * d/dw Z    where J is the cost function, S is the segmoid function, and Z is the linear regression
 after calculatin we get d/dw J = 1/n ∑(ŷ - y)X , which we put in the gradient 
 it gives W = W - r 1/n ∑ (ŷ - y)X    with x = 1 for w₀ 


## Evaluation of classification models (accuracy, precision, recall, F-measure) 24.10

Evaluation metrics for classification models are crucial for understanding the performance of the model and comparing different models. Here are the key metrics:

- **Accuracy**: The proportion of correct predictions among all predictions made by the model. It is calculated as `(TP + TN) / (TP + TN + FP + FN)`, where TP is true positives, TN is true negatives, FP is false positives, and FN is false negatives .

- **Precision**: The proportion of true positive predictions among all positive predictions made by the model. It is calculated as `TP / (TP + FP)`, where TP is true positives and FP is false positives .

- **Recall (Sensitivity)**: The proportion of true positive predictions among all actual positive instances in the data. It is calculated as `TP / (TP + FN)`, where TP is true positives and FN is false negatives .

- **F1-Score**: The harmonic mean of precision and recall, providing a balanced measure between them. It is calculated as `2 * (PPV * TPR) / (PPV + TPR)`, where PPV is the positive predictive value (precision) and TPR is the true positive rate (recall) .

- **Confusion Matrix**: A table that shows the number of true positive, false positive, true negative, and false negative predictions made by the model .

These metrics are particularly important because they provide a comprehensive view of the model's performance. Accuracy gives a general measure of how well the model is performing, while precision and recall give more insight into the model's performance on the positive class. The F1-score is particularly useful when you want a single metric that balances precision and recall, and the confusion matrix provides a detailed breakdown of the model's performance.

In practice, you would choose the metric that best aligns with the objectives of your model. For example, if it's more important to correctly identify as many positive cases as possible, you might prioritize recall. If it's more important to avoid classifying negative cases as positive, you might prioritize precision. The F1-score is often used when you want a balance between precision and recall .




## Linear regression


Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables. It is a fundamental machine learning algorithm that is widely used for many years due to its simplicity, interpretability, and efficiency .

Here's how linear regression works:

1. **Modeling the Relationship**: Linear regression assumes that there is a linear relationship between the dependent and independent variables. This means that the dependent variable can be predicted from the independent variables using a linear equation.

2. **Estimating Parameters**: The parameters of the linear regression model, often called coefficients, are estimated from the data. These coefficients represent the change in the dependent variable for a one-unit change in the independent variable(s).

3. **Predicting**: Once the model is trained, it can be used to predict the dependent variable for new data points based on the independent variables.

4. **Evaluation**: The performance of the linear regression model can be evaluated using various metrics such as the mean squared error (MSE), which measures the average squared difference between the actual and predicted values.

Linear regression is commonly used for predicting numerical values based on input features, forecasting future trends based on historical data, identifying correlations between variables, and understanding the impact of different factors on a particular outcome .

Here's an example of how to perform linear regression using Python's scikit-learn library:

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Assume X and y are your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
```

In this example, a linear regression model is trained on a dataset, and the mean squared error of the predictions is calculated .

Linear regression is relatively robust to outliers compared to other machine learning algorithms, and it often serves as a good baseline model for comparison with more complex machine learning algorithms. However, it assumes a linear relationship between the variables, which may not always hold true. When these assumptions are not met, other models may be more appropriate .

the linear regression assumes that there is a linear relation between the dependent variable (the prediction y) and the independent paramters (features xi) of type
 y = B₀ + B₁x₁ + B₂x₂ + .... + Bn xn + e
 where Bi are the coefficients and e is the error value

the cost function usually used is MSE  JB(X) = 1/n ∑(ŷ(X) - y)² and the task is to find the optimal B coefficient to minimize the error by reducing this function
we can use the gradient descent to do so 
Bj = Bj - a d/dB J(X)  where d/dBj of J is the partial derivative of J with respect to Bj for the cost function J
doing so will give the following 
Bj = Bj - a ∑=i(ŷ(Xi) - yi)xij   a is the learning rate, and xij is the value of jth (const j for Bj) feature for the ith data row (i in of the sum)


## Regularization, lasso and ridge regressions

Regularization is a technique used in machine learning to prevent overfitting by penalizing complex models. Overfitting occurs when a model learns the noise and randomness in the training data, which can lead to poor performance on unseen data. Regularization helps to prevent models from over-interpreting the training data by adding a penalty term to the loss function that the model seeks to minimize .

**Lasso Regression (L1 Regularization)**:
- Lasso regression, short for Least Absolute Shrinkage and Selection Operator, adds a penalty equal to the absolute value of the weights associated with each feature variable.
- It encourages sparsity by forcing some coefficients to reduce their values until they eventually become zero, effectively performing feature selection.
- Lasso regression is particularly useful when you have a large number of features and you want to automatically select the most important ones .

J = J₀ + λ∑|w| , where J₀ is the original cost function and lambda is the strength of the regularization


**Ridge Regression (L2 Regularization)**:
- Ridge regression, also known as L2 regularization, adds a penalty equal to the square of the weights associated with each feature variable.
- It encourages all coefficients to reduce in size by an amount proportional to their values, shrinking large weights toward zero.
- Ridge regularization can be more effective than Lasso when there are many collinear variables because it prevents individual coefficients from becoming too large and overwhelming others .

J = J₀ + λ ∑w²

**Selecting Optimal Alpha Values**:
- The penalty term in both Lasso and Ridge regression is controlled by a parameter often denoted as α (alpha) or  λ (lambda).
- The optimal value of α depends on the specific dataset and can be determined using techniques like cross-validation, where the model's performance is evaluated on different subsets of the data to find the α that produces the best generalization performance .

**Combining Lasso and Ridge Regularization**:
- Elastic Net regularization is a combination of Lasso and Ridge regularization. It adds both L1 and L2 penalties to the loss function.
- This method can produce simpler models while still utilizing most or all of the available features, offering a balance between the two techniques .

In practice, the choice between Lasso, Ridge, or Elastic Net regularization depends on the specific problem and the characteristics of the data. Lasso is typically used when feature selection is important, Ridge is used when all features are important and there is a concern about multicollinearity, and Elastic Net is used when there is a mix of important and less important features, and when there is a need for a balance between feature selection and regularization .




## Artificial neural networks(05.12)

Artificial Neural Networks (ANNs), also known as neural nets or ANNs, are a type of machine learning model inspired by the structure and function of biological neural networks found in animal brains. ANNs are composed of interconnected nodes, or artificial neurons, which are organized into layers: an input layer, one or more hidden layers, and an output layer .

Here's how ANNs work:

- **Input Layer**: This is where the network receives data. The input nodes of ANNs receive input signals, which can be the feature values of a sample of external data, such as images or documents .

- **Hidden Layers**: These layers are where the actual computation takes place. The hidden layer nodes compute the input signals based on synaptic weights, which are the strengths of the connections between nodes. These weighted inputs generate an output through a transfer function, which is typically a non-linear function .

- **Output Layer**: This layer produces the final output of the network. The outputs of the final output neurons of the neural net accomplish the task, such as recognizing an object in an image .

- **Learning**: ANNs learn from experience by adjusting the weights of the connections between nodes based on the error of the output compared to the desired output. This process is typically done using gradient descent or other optimization algorithms .

ANNs are capable of learning and modeling non-linearities and complex relationships, which makes them powerful tools in computer science and artificial intelligence. They are used for tasks such as predictive modeling, adaptive control, and various applications in artificial intelligence, including speech recognition, image recognition, and more .

In Python, the `scikit-learn` library provides tools for creating and training ANNs, such as the `MLPClassifier` for classification tasks and the `MLPRegressor` for regression tasks .




## Neuron activation functions, selection for a specific task

Neuron activation functions, also known as transfer functions, are a crucial component of neural networks. They determine the output of a neuron given an input or set of inputs. The role of an activation function is to introduce non-linearity into the output of a neuron, which allows neural networks to learn from complex, non-linear data .

Here are some common types of activation functions:

- **Linear Functions**: These are the simplest activation functions. They are linear, meaning the output is directly proportional to the input. They do not introduce non-linearity and are not suitable for most neural network models  f(z) = z

- **Sigmoid Function**: This function maps any input value to a value between   0 and   1, making it useful for binary classification problems. It is an S-shaped curve that can be interpreted as the probability of a binary outcome
  f(z) = 1 / (1 + e^-z)  range [0;1]

- **Tanh Function**: Similar to the sigmoid function, but it maps the input values to a range between   1 and   1. It is zero-centered, which can be beneficial for certain types of problems.
  f(z) = (e^z + e^-z) / (e^z - e^z)  range [-1;1]

- **ReLU (Rectified Linear Unit)**: This function outputs the input directly if it is positive; otherwise, it outputs zero. It is the most commonly used activation function in deep learning models due to its computational efficiency and ability to mitigate the vanishing gradient problem.
  f(z) = max(0, z)

- **Leaky ReLU**: A variant of ReLU that allows small negative values when the input is less than zero, which can help with the vanishing gradient problem.
  f(z) = max(αz, z) where α is a small number usually 0.01

- **Softmax Function**: This function is typically used in the output layer of a neural network for multi-class classification problems. It converts a vector of real numbers into a vector of probabilities, so the probabilities sum up to   1.

Choosing the right activation function depends on the specific task and the nature of the data. For example, if you're working on a binary classification problem, you might choose the sigmoid function for the output layer. If you're dealing with a multi-class classification problem, you might choose the softmax function. For hidden layers, ReLU or its variants like Leaky ReLU are commonly used due to their computational efficiency and ability to mitigate the vanishing gradient problem .


   
## Error back propagation method(05.12 - until 22:06)

Backpropagation is a widely used algorithm for training neural networks. It efficiently computes the gradients of the loss function with respect to the parameters (weights and biases) of the network, allowing for iterative optimization using gradient-based optimization algorithms like stochastic gradient descent.

Here's an overview of the backpropagation algorithm:

1. **Forward Pass:**
   - The forward pass involves computing the predicted output of the neural network for a given input data point. This is done by propagating the input forward through the network, computing the weighted sum of inputs and applying the activation function at each neuron in each layer until the output is obtained.

2. **Loss Computation:**
   - Once the predicted output is obtained, the loss function is computed by comparing the predicted output to the true target value. The choice of loss function depends on the task, such as mean squared error for regression or cross-entropy loss for classification.

3. **Backward Pass (Error Backpropagation):**
   - The backward pass involves computing the gradients of the loss function with respect to the parameters of the network, starting from the output layer and propagating backward through the network.
   - At each layer, the gradients are computed using the chain rule of calculus. The gradient of the loss function with respect to the output of each neuron in the layer is computed, and then the gradients with respect to the parameters (weights and biases) of the layer are computed based on these neuron-level gradients.
   - The gradients are then used to update the parameters of the network using an optimization algorithm such as gradient descent or one of its variants.

4. **Parameter Update:**
   - Once the gradients have been computed for all parameters of the network, the parameters are updated using the optimization algorithm. The parameters are adjusted in the direction that minimizes the loss function, based on the computed gradients and a learning rate hyperparameter.

5. **Iterative Optimization:**
   - Steps 1-4 are repeated iteratively for a fixed number of iterations or until convergence, with the goal of minimizing the loss function and improving the performance of the network on the training data.

By iteratively updating the parameters of the network based on the gradients computed using backpropagation, neural networks can learn to approximate complex mappings between input and output data, effectively solving a wide range of machine learning tasks.


## Convolutional neural networks: convolution operation, pooling, channels, feature map

Convolutional Neural Networks (CNNs) are a class of deep learning models that are particularly effective at processing grid-like data, such as images. They are composed of several layers, each of which performs a specific task in the process of learning to recognize patterns in the input data .

Here are the key components of a CNN:

- **Convolutional Layer**: This layer applies a series of filters to the input data. Each filter is a small matrix (the kernel) that slides over the input data. At each position, the kernel is multiplied element-wise with the input data, and the results are summed to produce a single value in the output, which forms the feature map. The purpose of the convolutional layer is to detect specific patterns or features within the input data. The filters are learnable parameters that the network adjusts during training to detect the most relevant features .

- **Pooling Layer**: The pooling layer is used to downsample the feature maps produced by the convolutional layer. This helps to reduce the dimensionality of the data, making the model more computationally efficient and less sensitive to the location of features in the input. There are several types of pooling operations, including max pooling (which takes the maximum value in a window) and average pooling (which takes the average value in a window). Pooling layers also help to introduce translation invariance, meaning the model can recognize features regardless of their position in the input .

- **Channels**: In the context of images, channels refer to the depth of the image data. For example, a color image has three channels (red, green, and blue), while a grayscale image has only one channel. The number of channels in the input data should match the number of channels in the kernel used in the convolutional layer .

- **Feature Map**: A feature map is the output of a convolutional layer or a pooling layer. It represents the presence of features in the input data that the layer has learned to recognize. Feature maps are used as input to the next layer in the network .

CNNs typically start with a convolutional layer followed by a pooling layer, and this pattern is repeated throughout the network. The final layers of a CNN are usually fully connected layers that perform the final classification or regression task .











