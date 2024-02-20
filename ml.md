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

Clustering is an unsupervised machine learning approach that aims to divide unlabeled data into groups or clusters, where similar data points are placed in the same cluster. The goal of clustering is to identify data structures within a dataset without relying on pre-existing labels or categories. Clustering can be used to improve supervised learning algorithms by using cluster labels as independent variables [0].

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

Data processing is the method of collecting raw data and translating it into usable information. It is a crucial step in the data lifecycle, which involves several stages [0][2][3]:

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
- **Standardization**: Scale features to have zero mean and unit variance. This is useful for many machine learning algorithms that assume that all features are centered around zero and have the same variance [1].
   
  ```python
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  ```

- **Normalization**: Scale features to have unit norm. This is useful for algorithms that use a quadratic form, like the dot product, to quantify similarity. It's also the basis of the Vector Space Model used in text classification and clustering [1].
   
  ```python
  from sklearn.preprocessing import Normalizer
  normalizer = Normalizer()
  X_normalized = normalizer.fit_transform(X)
  ```

- **Min-Max Scaling**: Scale features to a given range, usually  0 to  1. This is useful when you need the scaled features to have a specific range for some reason, such as when you need to ensure that a feature doesn't dominate others [1].

  ```python
  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler()
  X_scaled = scaler.fit_transform(X)
  ```

- **Robust Scaling**: Use a similar method to standardization but uses the median and interquartile range instead of the mean and standard deviation. This is useful when the data contains outliers [1].

  ```python
  from sklearn.preprocessing import RobustScaler
  scaler = RobustScaler()
  X_scaled = scaler.fit_transform(X)
  ```

When choosing a scaling method, consider the characteristics of your data and the requirements of the machine learning algorithms you plan to use. For example, if your data contains many outliers, robust scaling might be a better choice than standardization. If you're using a kernel method that requires unit norm, normalization would be appropriate.





## Metric clustering methods, metrics (2nd lecture)

Metric clustering methods in machine learning involve grouping data points based on certain metrics or distances between data points. These methods are often used to find structures in the data that are not obvious from the raw data alone. Here are some common metrics used in clustering:

- **Rand Index**: Measures the similarity between two data clusterings. It is useful when ground-truth cluster labels are available for comparison [1].
- **Silhouette Score**: Evaluates how close each sample in one cluster is to the samples in the neighboring clusters. It ranges from -1 to  1, with higher scores indicating better-defined clusters [2][3].
- **Davies-Bouldin Index**: Measures the average similarity between each cluster and its most similar one. A lower score indicates better separation between clusters [2][3].
- **Calinski-Harabasz Index**: Also known as the score index, it calculates the ratio of between-cluster variation to within-cluster variance. Higher values suggest more distinct groups [2].
- **Adjusted Rand Index**: Compares the resemblance of genuine class labels to predicted cluster labels. It is useful when ground-truth class labels are available [2].
- **Mutual Information (MI)**: Measures the agreement between the true class labels and the predicted cluster labels, indicating how well the clustering solution captures the underlying structure in the data [2].
- **Dunn Index**: Measures the ratio between the distance between the clusters and the distance within the clusters. A high Dunn index indicates that the clusters are well-separated and distinct [3].
- **Jaccard Coefficient**: Measures the similarity between the clustering results and the ground truth, considering the number of data points in each cluster [3].

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

1. **Choose the Number of Clusters (K)**: The K-means algorithm requires you to specify the number of clusters beforehand. This can be done using methods like the Elbow Method or the Silhouette Score to determine the optimal number of clusters [0][3].

2. **Initialize Centroids**: Randomly assign the initial centroids. The K-means++ technique is often used to initialize centroids because it spreads out the initial centroids, reducing the chances of poor outcomes [0].

3. **Assign Data Points to Clusters**: Calculate the distance from each data point to each centroid. Assign each data point to the cluster with the nearest centroid [3].

4. **Update Centroids**: Recalculate the centroid of each cluster by taking the mean of all data points assigned to that cluster [3].

5. **Iterate**: Repeat steps  3 and  4 until the centroids do not change significantly or a maximum number of iterations is reached. The algorithm has converged when the assignments of data points to clusters no longer change [3].

6. **Evaluate Clusters**: Check the quality of the clusters by looking at the within-cluster variance and ensuring that the clusters are meaningful and representative of the data [3].

Key Concepts:
- **Cluster Centroids**: The centroids represent the center of each cluster and are updated in each iteration to minimize the total intra-cluster variance.
- **Inertia**: Also known as within-cluster sum of squares (WCSS), inertia measures the compactness of the clusters. It is calculated as the sum of squared distances between each data point and its assigned centroid. Lower inertia indicates better clustering.

Advantages of K-means Clustering:
- Simple and easy to implement.
- Scalable to large datasets.
- Efficient in terms of computational complexity.
- Works well with spherical clusters.

Keep in mind that K-means has some limitations:
- It is sensitive to the initial placement of centroids, which can lead to different results with different initializations [0].
- It assumes that clusters are spherical and of similar size, which may not always be the case in real-world data [0].
- It can be sensitive to the scale of the data and the presence of outliers, so it's important to normalize the features before clustering [0].

For data with complex cluster shapes or when the number of clusters is not known, other clustering algorithms like Spectral Clustering or DBSCAN might be more suitable [0].



## The c-means fuzzy clustering method (4th lecture)

The C-means (or Fuzzy C-means) clustering method is a variant of the K-means algorithm that allows for fuzzy membership of data points to clusters. Here are the steps to perform C-means clustering:

1. **Choose the Number of Clusters (C)**: Similar to K-means, you start by choosing the number of clusters you want to create.

2. **Initialize Membership Degrees**: Assign membership degrees (u_ij) randomly to each data point for being in the clusters. These degrees represent the degree to which a data point belongs to a particular cluster.

3. **Compute Centroids**: Calculate the centroid for each cluster, which is the center of the cluster.

4. **Update Membership Degrees**: For each data point, compute its membership degrees for each cluster based on the inverse distance to the cluster centroid. This step is iterative and continues until the algorithm has converged, which means the change in membership degrees between two iterations is less than a given sensitivity threshold (ε) [0].

μij​=∑k=1K​(dik​dij​​)m−12​1​
µij = 1 / (∑k( (Dij / dik)^(2/(m-1)) ))
where dij is the disstance between the the point i and the cluster j

5. **Iterate**: Repeat steps  3 and  4 until the centroids do not change significantly or a maximum number of iterations is reached [0].

6. **Evaluate Clusters**: Check the quality of the clusters by looking at the membership degrees and ensuring that the clusters are meaningful and representative of the data.

C-means differs from K-means in that it assigns each data point a probability of belonging to each cluster, rather than assigning it to a single cluster. This allows for more flexible clustering, especially when the data points are not clearly separated into distinct clusters [1].

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
axes[0].scatter(X[:,0], X[:,1], alpha=.1)
axes[1].scatter(X[:,0], X[:,1], c=fcm_labels, alpha=.1)
axes[1].scatter(fcm_centers[:,0], fcm_centers[:,1], marker="+", s=500, c='w')
plt.show()
```

This code snippet demonstrates how to generate synthetic data, fit a Fuzzy C-means model to it, and then visualize the resulting clusters [3].


## Density algorithm of spatial clustering with the presence of noise (DBSCAN) (4 lec)


DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm that groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions (whose nearest neighbors are too far away). It is particularly useful for datasets where the number of clusters is not known a priori and the clusters can have arbitrary shapes [1][2].

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

DBSCAN is a density-based clustering algorithm, which means it can discover clusters of arbitrary shape and is also capable of finding arbitrarily shaped clusters, even clusters that are entirely surrounded by, but not connected to, a different cluster [1][2].

DBSCAN has several advantages:
- It does not require the user to set the number of clusters a priori.
- It can find arbitrarily shaped clusters.
- It is mostly insensitive to the ordering of the data points in the dataset.
- It can find a cluster completely surrounded by (but not connected to) a different cluster.

However, DBSCAN also has some disadvantages:
- It is not entirely deterministic due to the processing order of data points.
- The quality of DBSCAN depends on the distance measure used.
- It may not perform well on datasets with large differences in densities, as the `MinPts` and `ε` parameters may not be chosen appropriately for all clusters.

DBSCAN is available in various programming languages and libraries, such as scikit-learn in Python, ELKI in Java, and R packages like `dbscan` and `fpc`. These implementations often come with optimizations like k-d tree support for Euclidean distance, which can significantly improve the algorithm's performance on large datasets [1].



## Hierarchical clustering (4th lecture)

Hierarchical clustering is a method of cluster analysis that seeks to build a hierarchy of clusters. It is a way of grouping data points based on their similarities, and it can be particularly useful when you do not know the number of clusters in advance. Hierarchical clustering can be either agglomerative (bottom-up) or divisive (top-down) [0][3][4].

**Agglomerative Hierarchical Clustering**:
- **Bottom-Up Approach**: Each data point starts in its own cluster.
- **Process**: Pairs of clusters are merged as one moves up the hierarchy, based on a chosen distance measure.
- **Stopping Criterion**: The algorithm stops when all data points are merged into a single cluster.
- **Visualization**: The result is often represented in a dendrogram, which is a tree-like diagram showing the sequences of merges [0][4].

**Divisive Hierarchical Clustering**:
- **Top-Down Approach**: All data points start in one cluster.
- **Process**: The cluster is split into smaller clusters recursively as one moves down the hierarchy.
- **Stopping Criterion**: The algorithm stops when each data point is in its own cluster.
- **Visualization**: The result is also represented in a dendrogram, but it is an inverted tree showing the sequences of splits [0][4].

Hierarchical clustering has several advantages:
- It can handle non-convex clusters and clusters of different sizes and densities.
- It can handle missing data and noisy data.
- It reveals the hierarchical structure of the data, which can be useful for understanding the relationships among the clusters [4].

The choice between agglomerative and divisive clustering depends on the specific problem and the nature of the data. Hierarchical clustering can be implemented in various programming languages using libraries such as SciPy in Python, R, and Weka [0].

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
- **Data Compression**: Dimensionality reduction can lead to data compression, which reduces the storage space required for the dataset [0][1][3].

There are several techniques for dimensionality reduction, including:

- **Principal Component Analysis (PCA)**: This linear transformation technique identifies the axes in the feature space along which the data varies the most and projects the data onto these axes.
- **Singular Value Decomposition (SVD)**: This factorization technique can be used to perform PCA. It decomposes a matrix into singular vectors and singular values.
- **Linear Discriminant Analysis (LDA)**: This technique is used for classification problems and aims to maximize the separability of two or more classes.
- **t-Distributed Stochastic Neighbor Embedding (t-SNE)**: This non-linear technique is particularly good for visualizing high-dimensional data in two or three dimensions.
- **Autoencoders**: These are neural networks that learn to encode data into a lower-dimensional space and then decode it back to the original space.
- **Uniform Manifold Approximation and Projection (UMAP)**: This is a non-linear dimensionality reduction technique that can be used for visualization and feature extraction.

It's important to note that while dimensionality reduction can be beneficial, it can also lead to loss of information. Therefore, it's crucial to choose the right technique and parameters for your specific dataset and problem



## The Principal Component Method (PCA) (3rd lecture

Principal Component Analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. This transformation is defined in such a way that the first principal component has the largest possible variance (that is, accounts for as much of the variability in the data as possible), and each succeeding component in turn has the highest variance possible under the constraint that it is orthogonal to the preceding components [0].

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

t-Distributed Stochastic Neighbor Embedding (t-SNE) is a non-linear dimensionality reduction technique that is particularly well-suited for visualizing high-dimensional data in a lower-dimensional space, typically two or three dimensions. It was developed by Laurens van der Maaten and Geoffrey Hinton in  2008 [0][1][2].

Here's how t-SNE works:

1. **Pairwise Similarities**: t-SNE computes the pairwise similarities between data points in the high-dimensional space using a Gaussian kernel. The similarity between two points is determined by the inverse of the Euclidean distance between them.

2. **Mapping to Lower Dimensions**: The algorithm then maps the high-dimensional data points onto a lower-dimensional space (e.g.,  2D or  3D) while preserving the pairwise similarities. This is done by minimizing the divergence between the probability distribution of the original high-dimensional data and the lower-dimensional representation.

3. **Optimization**: The optimization process involves iteratively adjusting the positions of the points in the lower-dimensional space to better match the pairwise similarities from the high-dimensional space. This is typically done using gradient descent.

4. **Iterative Process**: t-SNE is an iterative algorithm, and the quality of the resulting visualization can depend on the initial conditions and the convergence of the algorithm.

t-SNE is particularly useful for visualizing complex datasets because it can reveal clusters and non-linear relationships that might not be apparent in the original high-dimensional space. However, it is not deterministic, meaning that it can produce different results on different runs. This can be an advantage when you want to explore the data from different perspectives, but it can also make it less suitable for scenarios where reproducibility is important [0][1][2].

The algorithm is available in various programming languages and libraries, such as scikit-learn in Python, Rtsne in R, and ELKI, which includes a Barnes-Hut approximation for handling large datasets [1][4].

In Python, using scikit-learn, you can perform t-SNE as follows:

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=40, random_state=42)
X_train_tsne = tsne.fit_transform(X_train)

print(tsne.kl_divergence_)
```

This code snippet demonstrates how to fit a t-SNE model to a dataset and transform the data into two dimensions [2].

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

Kohonen's Self-organizing Maps (SOMs) are a type of artificial neural network used for clustering and dimensionality reduction of high-dimensional data. SOMs are trained using a competitive learning algorithm, which allows the network to learn a low-dimensional representation of the input space of the training samples, typically in a two-dimensional form [0][1][3][4].

Here's an overview of how SOMs work:

1. **Initialization**: The SOM is initialized with a grid of nodes, which are the neurons of the network. Each node has a weight vector that represents its position in the input space.

2. **Training**: For each input vector in the training set, the SOM searches for the node that is closest to the input vector in the input space. This node is called the Best Matching Unit (BMU).

3. **Competitive Learning**: The BMU and its neighboring nodes are updated to become more like the input vector. This is done by adjusting the weights of the nodes to move them closer to the input vector. The amount of adjustment is determined by the learning rate and the distance to the input vector.

4. **Iterative Process**: The training process is repeated for all input vectors, and the nodes adjust their weights over time to form a map of the input space.

5. **Topological Preservation**: SOMs are designed to preserve the topological properties of the input space. This means that input vectors that are close to each other in the input space will also be close to each other in the SOM.

6. **Visualization**: The resulting map can be visualized as a two-dimensional grid, where each node represents a cluster of input vectors. This allows for the visualization and analysis of high-dimensional data in a more manageable way [0][1][3][4].

SOMs are particularly useful for visualizing complex datasets and for finding clusters in the data. They can be used for tasks such as image segmentation, document clustering, and anomaly detection. SOMs have been implemented in various programming languages and libraries, such as MATLAB, Python (with the `somoclu` package), and R (with the `kohonen` package) [1][3][4].

# Classification

## K-nearest neighbor classification method

The K-nearest neighbor (K-NN) classification method is a non-parametric, supervised learning algorithm that classifies new data points based on the majority class label of its nearest neighbors. Here's how the K-NN algorithm works for classification:

1. **Select the Number of Neighbors (K)**: Choose the number of neighbors to consider for classification. This is usually a small integer.

2. **Calculate Distances**: For a new data point, calculate the distance to all points in the training set.

3. **Find Nearest Neighbors**: Identify the K training points that are closest to the new data point.

4. **Vote for the Class**: Assign the new data point to the class that is most common among its K nearest neighbors.

5. **Predict**: The new data point is classified into the majority class of its K nearest neighbors.

The K-NN algorithm is considered a lazy learning method because it does not learn a model from the training data but instead stores the training dataset. When a new data point is presented for classification, it calculates the distance to all points in the training set, selects the K nearest neighbors, and assigns the new data point to the class that is most common among its neighbors [0][2][4].

The choice of K is critical, as a small K value will be sensitive to noise, while a large K value may oversmooth the boundaries. There are various methods to select an optimal K, such as cross-validation, which involves partitioning the dataset into subsets and evaluating the K-NN performance on each subset [3].

The K-NN algorithm can be used for both classification and regression tasks. In regression, the output is the average of the values of the K nearest neighbors [2].

In Python, the K-NN algorithm can be implemented using libraries like scikit-learn, which provides a KNeighborsClassifier class for classification and a KNeighborsRegressor class for regression [4].

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

In this example, the K-NN classifier is trained on the iris dataset with K set to  3, and then it is used to make predictions on the test set [4].










