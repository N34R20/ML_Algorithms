# Machine Learning Pipelines

In this lesson we’re going to learn how to turn a machine learning (ML) workflow to a pipeline using `scikit-learn`. A ML pipeline is a modular sequence of objects that codifies and automates a ML workflow to make it efficient, reproducible and generalizable. While the process of building pipelines is not singular, there are some tools that are universally used to do this. The most accessible of these is `scikit-learn`‘s `Pipeline` object which allows us to chain together the different steps that go into a ML workflow.

Turning a workflow into a pipeline has many other advantages too. Pipelines provide consistency — the same steps will always be applied in the same order under the same conditions. They also are very concise and can streamline your code. The `Pipeline` object within `scikit-learn` has consistent methods to use the many other estimators and transformers we have already covered in our ML curriculum. It is usually the starting point for a Machine Learning Engineer before turning to more sophisticated tools for scaling pipelines (such as PySpark, etc) and we will delve deeper into it in this lesson

What can go into a pipeline? For any of the intermediate steps, it must have both the `.fit` and `.transform` methods. This includes preprocessing, imputation, feature selection and dimensionality reduction. The final step must have the `.fit` method. Examples of tasks we’ve seen already that could benefit from a pipeline include:

- scaling data then applying principal component analysis
- filling in missing values then fitting a regression model
- one-hot-encoding categorical variables and scaling numerical variables

## **Data Cleaning (Numeric)**

To introduce pipelines, let’s look at a common task – dealing with missing values and scaling numeric variables. We will convert an existing code base to a pipeline, describing these two steps in detail.

To define a pipeline, pass a list of tuples of the form `(name, transform/estimator)` into the `Pipeline` object. For example, to use a `SimpleImputer` first, named “imputer”, and a `StandardScaler` second, named “scale”, pass these as as `Pipeline([("imputer",SimpleImputer()), ("scale",StandardScaler())])`. Once the pipeline has been instantiated, methods `.fit` and `.transform` can be called as before. If the last step of the pipeline is a model (i.e. has a `.predict` method), then this can also be called.

Each step in the pipeline will be fit in the order provided. Further parameters can be passed to each step as well. For example, if we want to pass the parameter `with_mean=False` to the `StandardScaler`, use `Pipeline([("imputer",SimpleImputer()), ("scale",StandardScaler(with_mean=False))])`.

```python
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

columns = ["sex","length","diam","height","whole","shucked","viscera","shell","age"]
df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",names=columns)

y = df.age
X=df.drop(columns=['age'])
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(include=['object']).columns
#create some missing values
for i in range(1000):
    X.loc[np.random.choice(X.index),np.random.choice(X.columns)] = np.nan

x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.25)

x_train_num = x_train[num_cols]
#fill missing values with mean on numeric features only
x_train_fill_missing = x_train_num.fillna(x_train_num.mean())
#fit standard scaler on x_train_fill_missing
scale = StandardScaler().fit(x_train_fill_missing)
#scale data after filling in missing values
x_train_fill_missing_scale = scale.transform(x_train_fill_missing)

#Now want to do the same thing on the test set! 
x_test_fill_missing = x_test[num_cols].fillna(x_train_num.mean())
x_test_fill_missing_scale = scale.transform(x_test_fill_missing)

#1. Rewrite using Pipelines!
pipeline = Pipeline([("imputer",None), ("scale",None)])
pipeline.fit(x_train[num_cols])

#2. Fit pipeline on the test and compare results
print('Verify pipeline transform test set is the same\nPrinting the sum of absolute differences:')
print(abs(x_test_fill_missing_scale - pipeline.transform(x_test[num_cols])).sum())

#3. Change imputer strategy to median and compare results
pipeline_median = Pipeline([("imputer",None), ("scale",None)])
pipeline_median.fit(x_train[num_cols])

print('Verify median pipeline transform is different\nPrinting the sum of absolute differences:')
print(abs(pipeline_median.transform(x_test[num_cols]) - pipeline.transform(x_test[num_cols])).sum())
```

## **Data Cleaning (Categorical)**

For the categorical variables, let’s look at another common task – dealing with missing values and one-hot-encoding. We will convert an existing codebase to a pipeline, describing the two steps in detail.

As in in the previous exercise, `SimpleImputer` will be used again to fill missing values in the pipeline, but this time, the strategy parameter will need to be updated to `most_frequent`. `OneHotEncoder` will be used as the second step in the pipeline. Note, that the default is that a sparse array will be returned from this transform, so we will use `sparse='False'` to return a full array.

## **Column Transformer**

Often times, you may not want to simply apply every function to all columns. If our columns are of different types, we may only want to apply certain parts of the pipeline to a subset of columns. This is what we saw in the two previous exercises. One set of transformations are applied to numeric columns and another set to the categorical ones. We can use `ColumnTransformer` as one way of combining these processes together.

`ColumnTransformer` takes in a list of tuples of the form `(name, transformer, columns)`. The transformer can be anything with a `.fit` and `.transform` method like we used previously (like `SimpleImputer` or `StandardScaler`), but can also itself be a pipeline, as we will use in the exercise.

```python
preprocess = ColumnTransformer(
    transformers=[
        ("cat_process", cat_vals, cat_cols),
        ("num_process", num_vals, num_cols)
    ]
)
```

## **Adding a Model**

Great! Now that we have all the preprocessing done and coded succinctly using `ColumnTransformer` and `Pipeline`, we can add a model. We will take the result at the end of the previous exercise, and now create a final pipeline with the `ColumnTransformer` as the first step, and a `LinearRegression` model as the second step.

By adding a model to the final step, the last step no longer has a `.transform` method. This is the only step in a pipeline that can be a non-transformer. But now the final step also has a `.predict` method, which can be called on the entire pipeline.

```python
import numpy as np
import pandas as pd

from sklearn import svm, datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
columns = ["sex","length","diam","height","whole","shucked","viscera","shell","age"]
df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",names=columns)

y = df.age
X=df.drop(columns=['age'])
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(include=['object']).columns
#create some missing values
for i in range(1000):
    X.loc[np.random.choice(X.index),np.random.choice(X.columns)] = np.nan

x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.25)

cat_vals = Pipeline([("imputer",SimpleImputer(strategy='most_frequent')), ("ohe",OneHotEncoder(sparse=False, drop='first'))])
num_vals = Pipeline([("imputer",SimpleImputer(strategy='mean')), ("scale",StandardScaler())])

preprocess = ColumnTransformer(
    transformers=[
        ("cat_process", cat_vals, cat_cols),
        ("num_process", num_vals, num_cols)
    ]
)
#1. Create a pipeline with pregrocess and a linear regression model
pipeline = Pipeline([("preprocess",preprocess), 
                     ("regr",LinearRegression())])

#2. Fit the pipeline on the training data
pipeline.fit(x_train, y_train)

#3. Predict the pipeline on the test data
print(pipeline.predict(x_test))
```

## **Hyperparameter Tuning**

Great, we have a very condensed bit of code that does all our data cleaning, preprocessing, and modeling in a reusable fashion! What now? Well, we can tune some of the parameters of the model by apply a grid search over a range of hyperparameter values.

A linear regression model has very few hyperparameters, really just whether we include in intercept. But we will use this as an example to see the process for a pipeline. The pipeline created in the previous exercise is, itself, an estimator – you can call `.fit` and `.predict` on it. So in fact, the pipeline can be passed as an estimator for `GridSearchCV`. This will then refit the pipeline for each combination of parameter values in the grid and each fold in the cross-validation split.

That’s a lot – but the code is again very short. One thing to keep in mind, to reference hyperparameters in a pipeline, the values are reference by the pipeline step name + ‘**‘ + hyperparameter. So `regr**fit_intercept` references the named pipeline step “regr” and the hyperparameter “fit_intercept”.

```python
import numpy as np
import pandas as pd

from sklearn import svm, datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn import metrics

columns = ["sex","length","diam","height","whole","shucked","viscera","shell","age"]
df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",names=columns)

y = df.age
X=df.drop(columns=['age'])
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(include=['object']).columns
#create some missing values
for i in range(1000):
    X.loc[np.random.choice(X.index),np.random.choice(X.columns)] = np.nan

x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.25)

cat_vals = Pipeline([("imputer",SimpleImputer(strategy='most_frequent')), ("ohe",OneHotEncoder(sparse=False, drop='first'))])
num_vals = Pipeline([("imputer",SimpleImputer(strategy='mean')), ("scale",StandardScaler())])

preprocess = ColumnTransformer(
    transformers=[
        ("cat_process", cat_vals, cat_cols),
        ("num_process", num_vals, num_cols)
    ]
)

#Create a pipeline with pregrocess and a linear regression model
pipeline = Pipeline([("preprocess",preprocess), 
                     ("regr",LinearRegression())])

#Very simple parameter grid, with and without the intercept
param_grid = {
    "regr__fit_intercept": [True,False]
}
#1. Grid search using previous pipeline
gs = GridSearchCV(pipeline, param_grid, scoring='neg_mean_squared_error', cv=5)

#2. fit grid using training data and print best score
gs.fit(x_train, y_train)
print('Best NMSE:')
print(gs.best_score_)
```

## **Final Pipeline**

Way to go! Now that we are getting the hang of pipelines, let’s take things up a notch and now search over a range of different types of models, all of which have their own sets of hyperparameters. In the original `pipeline`, we defined `regr` to be an instance of `LinearRegression`. Then in defining the parameter grid to search over, we used the dictionary `{"regr__fit_intercept": [True,False]}` to define the values of the `fit_intercept` term. We can equivalently do this by passing both the estimator AND parameters in a dictionary as

`{'regr': [LinearRegression()], "regr__fit_intercept": [True,False]}`

There are two main ways we can access elements of this pipeline:

1. We can access the list of steps by using `.steps` and accessing the elements of the tuple. This of the same form as how the pipeline was initially defined. For examples, to access the last step of `best_model` use `best_model.steps[-1]`. This will return a tuple with the name and estimator. To access the estimator, use `best_model.steps[-1][-1]`.
2. Another way the steps can be accessed is by name – after all, we gave each steps a string name when we defined them. The same regression model can be access using `best_model.named_steps['regr']` to access the named step `regr`.

```python
# 1. Update the search_space dictionary to include values for alpha in lasso and ridge regression models. Use np.logspace(-4,2,10).
search_space = [{'regr': [LinearRegression()]}, # Actual Estimator
                {'regr':[Ridge()],
                     'regr__alpha': np.logspace(-4, 2, 10)},
                {'regr':[Lasso()],
                     'regr__alpha': np.logspace(-4, 2, 10)}]

# 2.  Fit the GridSearchCV on the training data and print the best estimator and score from the search.
gs = GridSearchCV(pipeline, search_space, scoring='neg_mean_squared_error', cv=5)
gs.fit(x_train, y_train)
print('Best Estimator:')
print(gs.best_estimator_)
print('Best NMSE:')
print(gs.best_score_)

#3. Save the best estimator and print it
best_model = gs.best_estimator_
print('The regression model is:')
print(best_model.named_steps['regr'])
print('The hyperparameters of the regression model are:')
print(best_model.named_steps['regr'].get_params())

#4. Access the hyperparameters of the categorical preprocessing step

print('The hyperparameters of the imputer are:')
print(best_model.named_steps['preprocess'].named_transformers_['cat_process'].named_steps['imputer'].get_params())
```

## **Writing Custom Classes & Summary**

While scikit-learn contains many existing transformers and classes that can be used in pipelines, you may need at some point to create your own. This is simpler than you may think, as a step in the pipeline needs to have only a few methods implemented. If it is an intermediate step, it will need fit and transform methods, which we will demonstrate in the exercise below.

Here are some of the major takeaways on pipeline:

- Pipelines help make concise, reproducible, code by combining steps of transformers and/or a final estimator.
- Intermediate steps of a pipeline must have both the `.fit()` and `.transform()` methods. This includes preprocessing, imputation, feature selection, dimension reduction.
- The final step of a pipeline must have the `.fit()` method – this can include a transformer or an estimator/model.
- If the pipeline is meant to only transform your data by combining preprocessing and data cleaning steps, then each step in the pipeline will be a transformer. If your pipeline will also include a model (a final estimation or prediction step), then the last step must be an estimator.
- Once the steps of a pipeline are defined, it can be used like an other transformer/estimator by calling fit, transform, and/or predict methods. Similarly, it can be used in place of an estimator in a hyperparameter grid search.

```python
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

columns = ["sex","length","diam","height","whole","shucked","viscera","shell","age"]
df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",names=columns)
y = df.age
X=df.drop(columns=['age'])
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(include=['object']).columns

for i in range(1000):
    X.loc[np.random.choice(X.index),np.random.choice(X.columns)] = np.nan
x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.25)
x_train_num = x_train[num_cols]
#fill missing values with mean on numeric features only
x_train_fill_missing = x_train_num.fillna(x_train_num.mean())
#fit standard scaler on x_train_fill_missing
scale = StandardScaler().fit(x_train_fill_missing)
#scale data after filling in missing values
x_train_fill_missing_scale = scale.transform(x_train_fill_missing)
x_test_fill_missing = x_test[num_cols].fillna(x_train_num.mean())
x_test_fill_missing_scale = scale.transform(x_test_fill_missing)

class MyImputer(BaseEstimator, TransformerMixin): 
    def __init__(self):
        return None
    
    def fit(self, X, y = None):
        self.means = np.mean(X, axis=0)    # calculate the mean of each column
        return self
    
    def transform(self, X, y = None):
        #transform method fills in missing values with means using pandas
        return X.fillna(self.means)

#1. Create new pipeline using the custom class MyImputer as the first step and standard scaler on the second
new_pipeline = Pipeline([("imputer",MyImputer()), ("scale",StandardScaler())])

#2. 1.Fit new pipeline on the training data with num_cols only
new_pipeline.fit(x_train[num_cols])
x_transform = new_pipeline.transform(x_test[num_cols])

#2 2. Verify that the results of the transform are the same on test set
array_diff = abs(x_transform - x_test_fill_missing_scale).sum()
print(array_diff)
```