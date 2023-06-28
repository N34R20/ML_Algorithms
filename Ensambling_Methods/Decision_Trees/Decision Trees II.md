# Decision Trees

## **What is a Decision Tree?**

Decision trees are machine learning models that try to find patterns in the features of data points. Take a look at the tree on this page. This tree tries to predict whether a student will get an A on their next test.

By asking questions like “What is the student’s average grade in the class” the decision tree tries to get a better understanding of their chances on the next test.

In order to make a classification, this classifier needs a data point with four features:

- The student’s average grade in the class.
- The number of hours the student plans on studying for the test.
- The number of hours the student plans on sleeping the night before the test.
- Whether or not the student plans on cheating.

For example, let’s say that somebody has a “B” average in the class, studied for more than 3 hours, slept less than 5 hours before the test, and doesn’t plan to cheat. If we start at the top of the tree and take the correct path based on that data, we’ll arrive at a *leaf node* that predicts the person will *not* get an A on the next test.

## **What is a Decision Tree? (Contd.)**

If we’re given this magic tree, it seems relatively easy to make classifications. But how do these trees get created in the first place? Decision trees are supervised machine learning models, which means that they’re created from a training set of labeled data. Creating the tree is where the *learning* in machine learning happens.

Take a look at the gif on this page. We begin with every point in the training set at the top of the tree. These training points have labels — the red points represent students that didn’t get an A on a test and the green points represent students that did get an A on a test.

We then decide to split the data into smaller groups based on a feature. For example, that feature could be something like their average grade in the class. Students with an A average would go into one set, students with a B average would go into another subset, and so on.

Once we have these subsets, we repeat the process — we split the data in each subset again on a different feature. Eventually, we reach a point where we decide to stop splitting the data into smaller groups. We’ve reached a leaf of the tree. We can now count up the labels of the data in that leaf. If an unlabeled point reaches that leaf, it will be classified as the majority label.

We can now make a tree, but how did we know which features to split the data set with? After all, if we started by splitting the data based on the number of hours they slept the night before the test, we’d end up with a very different tree that would produce very different results. How do we know which tree is best? We’ll tackle this question soon!

## **Implementing a Decision Tree**

To answer the questions posed in the previous exercise, we’re going to do things a bit differently in this lesson and work “backwards” (!!!): we’re going to first fit a decision tree to a dataset and visualize this tree using `scikit-learn`. We’re then going to systematically unpack the following: how to interpret the tree visualization, how `scikit-learn`‘s implementation works, what is gini impurity, what are parameters and hyper-parameters of the decision tree model, etc.

We’re going to use a dataset about cars with six features:

- The price of the car, `buying`, which can be “vhigh”, “high”, “med”, or “low”.
- The cost of maintaining the car, `maint`, which can be “vhigh”, “high”, “med”, or “low”.
- The number of doors, `doors`, which can be “2”, “3”, “4”, “5more”.
- The number of people the car can hold, `persons`, which can be “2”, “4”, or “more”.
- The size of the trunk, `lugboot`, which can be “small”, “med”, or “big”.
- The safety rating of the car, `safety`, which can be “low”, “med”, or “high”.

The question we will be trying to answer using decision trees is: when considering buying a car, what factors go into making that decision?

1

- Take a look at the first five rows of the dataset by uncommenting `print(df.head())` and clicking Run.
- We’ve created dummy features for the categorical values and set the predictor and target variables as `X` and `y` respectively. Uncomment the lines pertaining to this and press Run.
- You can examine the new set of features using `print(X.columns)`

```python
#Import models from scikit learn module:
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#Loading the dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'accep'])

## 1a. Take a look at the dataset
print(df.head())

## 1b. Setting the target and predictor variables
df['accep'] = ~(df['accep']=='unacc') #1 is acceptable, 0 if not acceptable
X = pd.get_dummies(df.iloc[:,0:6])
y = df['accep']

## 1c. Examine the new features
print(X.columns)
print(len(X.columns))
```

We can now perform a train-test split and fit a decision tree to our training data. We’ll be using `scikit-learn`
‘s `train_test_split`
 function to do the split and the `DecisionTreeClassifier()`
 class to fit the data.

```python
## 2a. Performing the train-test split
x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.2)

## 2b.Fitting the decision tree classifier
dt = DecisionTreeClassifier(max_depth=3, ccp_alpha=0.01,criterion='gini')
dt.fit(x_train, y_train)
```

We’re now ready to visualize the decision tree! The `tree`
 module within `scikit-learn`
 has a plotting functionality that allows us to do this.

```python
## 3.Plotting the Tree
plt.figure(figsize=(20,12))
tree.plot_tree(dt, feature_names = x_train.columns, max_depth=5, class_names = ['unacc', 'acc'], label='all', filled=True)
plt.tight_layout()
plt.show()
```

## **Interpreting a Decision Tree**

We’re now going to examine the decision tree we built for the car dataset. The image to the right of the code editor is the exact plot you created in the previous exercise. Two important concepts to note here are the following:

1. The root node is identified as the top of the tree. This is notated already with the number of samples and the numbers in each class (i.e. unacceptable vs. acceptable) that was used to build the tree.
2. Splits occur with True to the left, False to the right. Note the right split is a `leaf node` i.e., there are no more branches. Any decision ending here results in the majority class. (The majority class here is `unacc`.)

(Note that there is a term called `gini` in each of the boxes that is immensely important for how the split is done - we will explore this in the following exercise!)

To interpret the tree, it’s useful to keep in mind that the variables we’re looking at are categorical variables that correspond to:

- `buying`: The price of the car which can be “vhigh”, “high”, “med”, or “low”.
- `maint`: The cost of maintaining the car which can be “vhigh”, “high”, “med”, or “low”.
- `doors`: The number of doors which can be “2”, “3”, “4”, “5more”.
- `persons`: The number of people the car can hold which can be “2”, “4”, or “more”.
- `lugboot`: The size of the trunk which can be “small”, “med”, or “big”.
- `safety`: The safety rating of the car which can be “low”, “med”, or “high”.

## **Gini Impurity**

Consider the two trees below. Which tree would be more useful as a model that tries to predict whether someone would get an A in a class?

![https://content.codecademy.com/programs/data-science-path/decision-trees/comparison_1.svg](https://content.codecademy.com/programs/data-science-path/decision-trees/comparison_1.svg)

![https://content.codecademy.com/programs/data-science-path/decision-trees/comparison_2.svg](https://content.codecademy.com/programs/data-science-path/decision-trees/comparison_2.svg)

Let’s say you use the top tree. You’ll end up at a leaf node where the label is up for debate. The training data has labels from both classes! If you use the bottom tree, you’ll end up at a leaf where there’s only one type of label. There’s no debate at all! We’d be much more confident about our classification if we used the bottom tree.

This idea can be quantified by calculating the *Gini impurity* of a set of data points. For two classes (1 and 2) with probabilites `p_1` and `p_2` respectively, the Gini impurity is:

$$
1 - (p_1^2 + p_2^2) = 1 - (p_1^2 + (1-p_1)^2)

$$

![https://static-assets.codecademy.com/skillpaths/ml-fundamentals/decision_trees/gini_impurity_graph.png](https://static-assets.codecademy.com/skillpaths/ml-fundamentals/decision_trees/gini_impurity_graph.png)

The goal of a decision tree model is to separate the classes the best possible, i.e. minimize the impurity (or maximize the purity). Notice that if p_1

is 0 or 1, the Gini impurity is 0, which means there is only one class so there is perfect separation. From the graph, the Gini impurity is maximum at p_1=0.5, which means the two classes are equally balanced, so this is perfectly impure!

In general, the Gini impurity for C classes is defined as:

$$
1 - \sum_1^C p_i^2

$$

## **Information Gain**

We know that we want to end up with leaves with a low Gini Impurity, but we still need to figure out which features to split on in order to achieve this. To answer this question, we can calculate the *information gain* of splitting the data on a certain feature. Information gain measures the difference in the impurity of the data before and after the split.

For example, let’s start with the root node of our car acceptability tree:

![https://static-assets.codecademy.com/skillpaths/ml-fundamentals/decision_trees/dtree_info_gain.png](https://static-assets.codecademy.com/skillpaths/ml-fundamentals/decision_trees/dtree_info_gain.png)

The initial Gini impurity (which we confirmed previously) is 0.418. The first split occurs based on the feature `safety_low<=0.5`, and as this is a dummy variable with values 0 and 1, this split is pushing not low safety cars to the left (912 samples) and low safety cars to the right (470 samples). Before we discuss how we decided to split on this feature, let’s calculate the information gain.

The new Gini impurities for these two split nodes is 0.495 and 0 (which is a pure leaf node!). All together, the now weighted Gini impurity after the split is:

$$
912/1382*(.495) + 470/1382*(0) = 0.3267

$$

Not bad! (Remember we *want* our Gini impurity to be lower!) This is lower than our initial Gini impurity, so by splitting the data in that way, we’ve gained some information about how the data is structured — the datasets after the split are purer than they were before the split.

Then the information gain (or reduction in impurity after the split) is

$$
0.4185 - 0.3267 = 0.09180

$$

The higher the information gain the better — if information gain is 0, then splitting the data on that feature was useless!

## **How a Decision Tree is Built (Feature Split)**

Now that we know about Gini impurity and information gain, we can understand how the decision tree was built. We’ve been using an already built tree, where the root node is split on `safety_low<=0.5`, which is true when the value is 0, the left split (vehicles WITHOUT low safety), and false when the value is 1, the right split (vehicles WITH low safety). We saw in the previous exercise the information gain for this feature was

$$
0.4185 - 0.3267 = 0.0918.

$$

Now, let’s compare that with a different feature we could have split on first, `persons_2`. In this case, the left branch will have a Gini impurity of

$$
1 - (505/917)^2 - (412/917)^2 = 0.4949

$$

while the right split has a Gini impurity of 0. (Don’t worry about the numbers - you will be able to verify this presently!)

Then the weighted impurity is

$$
917/1382 (0.4949) + 465/1382 (0) = 0.3284

$$

The information gain then is

$$
0.4185 - 0.3284 = 0.0901

$$

Since this is less than the information gain from `safety_low`, this is not a better split and so it is not used. We would continue this process with ALL features, but, spoiler, `safety_low` gives the largest information gain, and so was chosen first. To verify that this is indeed the case, we have two functions pre-written in the workspace, `gini` and `info_gain` that calculate Gini impurity and information gain at any node.

```sql
## The usual libraries, loading the dataset and performing the train-test split
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'accep'])
df['accep'] = ~(df['accep']=='unacc') #1 is acceptable, 0 if not acceptable
X = pd.get_dummies(df.iloc[:,0:6])
y = df['accep']

x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.2)

## Functions to calculate gini impurity and information gain

def gini(data):
    """calculate the Gini Impurity
    """
    data = pd.Series(data)
    return 1 - sum(data.value_counts(normalize=True)**2)
   
def info_gain(left, right, current_impurity):
    """Information Gain associated with creating a node/split data.
    Input: left, right are data in left branch, right banch, respectively
    current_impurity is the data impurity before splitting into left, right branches
    """
    # weight for gini score of the left branch
    w = float(len(left)) / (len(left) + len(right))
    return current_impurity - w * gini(left) - (1 - w) * gini(right)

## 1. Calculate gini and info gain for a root node split at safety_low<=0.5
y_train_sub = y_train[x_train['safety_low']==0]
x_train_sub = x_train[x_train['safety_low']==0]

gi = gini(y_train_sub)
print(f'Gini impurity at root: {gi}')

## 2. Information gain when using feature `persons_2`
left = y_train[x_train['persons_2']==0]
right = y_train[x_train['persons_2']==1]

print(f'Information gain for persons_2: {info_gain(left, right, gi)}')

## 3. Which feature split maximizes information gain?

info_gain_list = []
for i in x_train.columns:
    left = y_train_sub[x_train_sub[i]==0]
    right = y_train_sub[x_train_sub[i]==1]
    info_gain_list.append([i, info_gain(left, right, gi)])

info_gain_table = pd.DataFrame(info_gain_list).sort_values(1,ascending=False)
print(f'Greatest impurity gain at:{info_gain_table.iloc[0,:]}')
print(info_gain_table)
```

## **How a Decision Tree is Built (Recursion)**

Now that we can find the best feature to split the dataset, we can repeat this process again and again to create the full tree. This is a recursive algorithm! We start with every data point from the training set, find the best feature to split the data, split the data based on that feature, and then recursively repeat the process again on each subset that was created from the split.

We’ll stop the recursion when we can no longer find a feature that results in any information gain. In other words, we want to create a leaf of the tree when we can’t find a way to split the data that makes purer subsets.

```sql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

def gini(data):
    """calculate the Gini Impurity
    """
    data = pd.Series(data)
    return 1 - sum(data.value_counts(normalize=True)**2)
   
def info_gain(left, right, current_impurity):
    """Information Gain associated with creating a node/split data.
    Input: left, right are data in left branch, right banch, respectively
    current_impurity is the data impurity before splitting into left, right branches
    """
    # weight for gini score of the left branch
    w = float(len(left)) / (len(left) + len(right))
    return current_impurity - w * gini(left) - (1 - w) * gini(right)

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'accep'])
df['accep'] = ~(df['accep']=='unacc') #1 is acceptable, 0 if not acceptable
X = pd.get_dummies(df.iloc[:,0:6])
y = df['accep']

x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.2)

y_train_sub = y_train[x_train['safety_low']==0]
x_train_sub = x_train[x_train['safety_low']==0]

gi = gini(y_train_sub)
print(f'Gini impurity at root: {gi}')

info_gain_list = []
for i in x_train.columns:
    left = y_train_sub[x_train_sub[i]==0]
    right = y_train_sub[x_train_sub[i]==1]
    info_gain_list.append([i, info_gain(left, right, gi)])

info_gain_table = pd.DataFrame(info_gain_list).sort_values(1,ascending=False)
print(f'Greatest impurity gain at:{info_gain_table.iloc[0,:]}')
```

## **Train and Predict using `scikit-learn`**

Now we’ll finally build a decision tree ourselves! We will use `scikit-learn`‘s tree module to create, train, predict, and visualize a decision tree classifier. The syntax is the same as other models in `scikit-learn`, so it should look very familiar. First, an instance of the model class is instantiated with `DecisionTreeClassifier()`. To use non-default hyperparameter values, you can pass them at this stage, such as `DecisionTreeClassifier(max_depth=5)`.

Then `.fit()` takes a list of data points followed by a list of the labels associated with that data and builds the decision tree model.

Finally, once we’ve made our tree, we can use it to classify new data points. The `.predict()` method takes an array of data points and will return an array of classifications for those data points. `predict_proba()` can also be used to return class probabilities instead. Last, `.score()` can be used to generate the accuracy score for a new set of data and labels.

As with other sklearn models, only numeric data can be used (categorical variables and nulls must be handled prior to model fitting).

```sql
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'accep'])
df['accep'] = ~(df['accep']=='unacc') #1 is acceptable, 0 if not acceptable
X = pd.get_dummies(df.iloc[:,0:6])
y = df['accep']
x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.2)

## 1. Create a decision tree and print the parameters
dtree = DecisionTreeClassifier()
print(f'Decision Tree parameters: {dtree.get_params()}')

## 2. Fit decision tree on training set and print the depth of the tree
dtree.fit(x_train, y_train)
print(f'Decision tree depth: {dtree.get_depth()}')

## 3. Predict on test data and accuracy of model on test set
y_pred = dtree.predict(x_test)

print(f'Test set accuracy: {dtree.score(x_test, y_test)}') # or accuracy_score(y_test, y_pred)
```

## **Visualizing Decision Trees**

Great, we built a decision tree using `scikit-learn` and predicted new values with it! But what does the tree look like? What features are used to split? Two methods using only `scikit-learn`/`matplotlib` can help visualize the tree, the first using `tree_plot`, the second listing the rules as text. There are other libraries available with more advanced visualization (`graphviz` and `dtreeviz`, for example, but may require additional installation and won’t be covered here).

```sql
import codecademylib3
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

## Loading the data and setting target and predictor variables
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'accep'])
df['accep'] = ~(df['accep']=='unacc') #1 is acceptable, 0 if not acceptable
X = pd.get_dummies(df.iloc[:,0:6])
y = df['accep']

## Train-test split and fitting the tree
x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.3)
dtree = DecisionTreeClassifier(max_depth=3)
dtree.fit(x_train, y_train)

## Visualizing the tree
plt.figure(figsize=(27,12))
tree.plot_tree(dtree)
plt.tight_layout()
plt.show()

## Text-based visualization of the tree (View this in the Output terminal!)
print(tree.export_text(dtree))
```

## **Advantages and Disadvantages**

As we have seen already, decision trees are easy to understand, fully explainable, and have a natural way to visualize the decision making process. In addition, often little modification needs to be made to the data prior to modeling (such as scaling, normalization, removing outliers) and decision trees are relatively quick to train and predict. However, now let’s talk about some of their limitations.

One problem with the way we’re currently making our decision trees is that our trees aren’t always *globally optimal*. This means that there might be a better tree out there somewhere that produces better results. But wait, why did we go through all that work of finding information gain if it’s not producing the best possible tree?

Our current strategy of creating trees is *greedy*. We assume that the best way to create a tree is to find the feature that will result in the largest information gain *right now* and split on that feature. We never consider the ramifications of that split further down the tree. It’s possible that if we split on a suboptimal feature right now, we would find even better splits later on. Unfortunately, finding a globally optimal tree is an extremely difficult task, and finding a tree using our greedy approach is a reasonable substitute.

Another problem with our trees is that they are prone to *overfit* the data. This means that the structure of the tree is too dependent on the training data and may not generalize well to new data. In general, larger trees tend to overfit the data more. As the tree gets bigger, it becomes more tuned to the training data and it loses a more generalized understanding of the real world data.