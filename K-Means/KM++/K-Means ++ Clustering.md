# K-Means ++ Clustering

## **Introduction to K-Means++**

The K-Means clustering algorithm is more than half a century old, but it is not falling out of fashion; it is still the most popular clustering algorithm for Machine Learning.

However, there can be some problems with its first step. In the traditional K-Means algorithms, the starting postitions of the centroids are intialized completely randomly. This can result in suboptimal clusters.

In this lesson, we will go over another version of K-Means, known as the **K-Means++ algorithm**. K-Means++ changes the way centroids are initalized to try to fix this problem.

## **Poor Clustering**

Suppose we have four data samples that form a rectangle whose width is greater than its height:

![https://content.codecademy.com/programs/machine-learning/k-means-plus-plus/no_clusers.png](https://content.codecademy.com/programs/machine-learning/k-means-plus-plus/no_clusers.png)

If you wanted to find two clusters (`k` = 2) in the data, which points would you cluster together? You might guess the points that align vertically cluster together, since the height of the rectangle is smaller than its width. We end up with a left cluster (purple points) and a right cluster (yellow points).

![https://content.codecademy.com/programs/machine-learning/k-means-plus-plus/correct.png](https://content.codecademy.com/programs/machine-learning/k-means-plus-plus/correct.png)

Let’s say we use the regular K-Means algorithm to cluster the points, where the cluster centroids are initialized randomly. We get unlucky and those randomly initialized cluster centroids happen to be the midpoints of the top and bottom line segments of the rectangle formed by the four data points.

![https://content.codecademy.com/programs/machine-learning/k-means-plus-plus/poor_cluster.png](https://content.codecademy.com/programs/machine-learning/k-means-plus-plus/poor_cluster.png)

The algorithm would converge immediately, without moving the cluster centroids. Consequently, the two top data points are clustered together (yellow points) and the two bottom data points are clustered together (purple points).

![https://content.codecademy.com/programs/machine-learning/k-means-plus-plus/final_cluster.png](https://content.codecademy.com/programs/machine-learning/k-means-plus-plus/final_cluster.png)

This is a *suboptimal clustering* because the width of the rectangle is greater than its height. The optimal clusters would be the two left points as one cluster and the two right points as one cluster, as we thought earlier.

## **What is K-Means++?**

To recap, the Step 1 of the K-Means algorithm is “Place `k` random centroids for the initial clusters”.

The K-Means++ algorithm replaces Step 1 of the K-Means algorithm and adds the following:

- **1.1** The first cluster centroid is randomly picked from the data points.
- **1.2** For each remaining data point, the distance from the point to its nearest cluster centroid is calculated.
- **1.3** The next cluster centroid is picked according to a probability proportional to the distance of each point to its nearest cluster centroid. This makes it likely for the next cluster centroid to be far away from the already initialized centroids.

**Repeat 1.2 - 1.3** until `k` centroids are chosen.

![Captura de Pantalla 2022-12-11 a la(s) 19.43.34.png](K-Means%20++%20Clustering%20aee046f24fdc4b84a9bfa887715f6c77/Captura_de_Pantalla_2022-12-11_a_la(s)_19.43.34.png)

![Captura de Pantalla 2022-12-11 a la(s) 19.43.47.png](K-Means%20++%20Clustering%20aee046f24fdc4b84a9bfa887715f6c77/Captura_de_Pantalla_2022-12-11_a_la(s)_19.43.47.png)

## **K-Means++ using Scikit-Learn**

Using the [scikit-learn](http://scikit-learn.org/stable/) library and its `cluster` module , you can use the `KMeans()` method to build an original K-Means model that finds 6 clusters like so:

```python
model = KMeans(n_clusters=6, init='random')

```

The `init` parameter is used to specify the initialization and `init='random'` specifies that initial centroids are chosen as random (the original K-Means).

But how do you implement K-Means++?

There are two ways and they both require little change to the syntax:

**Option 1:** You can adjust the parameter to `init='k-means++'`.

```python
test = KMeans(n_clusters=6, init='k-means++')

```

**Option 2:** Simply drop the parameter.

```python
test = KMeans(n_clusters=6)

```

This is because that `init=k-means++` is actually *default* in scikit-learn.

## **Review**

Congratulations, now your K-Means model is improved and ready to go!

K-Means++ improves K-Means by placing initial centroids more strategically. As a result, it can result in more optimal clusterings than K-Means.

It can also outperform K-Means in speed. If you get very unlucky initial centroids using K-Means, the algorithm can take a long time to converge. K-Means++ will often converge quicker!

You can implement K-Means++ with the [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) library similar to how you implement K-Means.

The `KMeans()` function has an `init` parameter, which specifies the method for initialization:

- `'random'`
- `'k-means++'`

**Note:** scikit-learn’s `KMeans()` uses `'k-means++'` by default, but it is a good idea to be explicit!