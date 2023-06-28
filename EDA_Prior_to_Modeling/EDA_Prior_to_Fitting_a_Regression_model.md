# EDA Prior to Fitting a Regression Model

# **Introduction**

Before fitting any model, it is often important to conduct EDA in order to check assumptions, inspect the data for anomalies (such as missing, duplicated, or mis-coded data), and inform feature selection/transformation. In this article, we will explore some of the EDA techniques that are generally employed prior to fitting a regression model.

# **The data**

For our example analysis, we’ve downloaded a dataset from [Kaggle](https://www.kaggle.com/cyaris/2016-mlb-season?select=baseball_reference_2016_clean.csv) which contains data on MLB games from the 2016 season. We’ve saved this data as a dataframe named **`bb`**. Suppose that we want to fit a linear regression to predict **`attendance`** using the following predictors:

- **`game_type`** — is the game during the day or at night
- **`day_of_week`** — what day of the week did the game occur
- **`temperature`** — average game temperature (Fahrenheit)
- **`sky`** — description of sky condition at the time of the game
- **`total_runs`** — total runs scored in the game

# **Preview the dataset**

Any EDA process will probably begin by inspecting a subset of data. For a Pandas dataframe, this can be done by using the **`.head()`** method:

```
bb.head()

```

[Untitled Database](https://www.notion.so/6adbea2082144bb3bb6a44ab18284f7f?pvs=21)

By looking at the first few rows of the data, we can often figure out what kind of data we have (eg., discrete or continuous) and get a sense of how they are coded. For example, we can see that **`attendance`**, **`temperature`**, and **`total_runs`** are numbers, while **`game_type`**, **`day_of_week`**, and **`sky`** appear to be text.

After our initial inspection, we’ll want to dig deeper to investigate the following:

- the data type of each variable
- how discrete/categorical data is coded (and whether we need to make any changes)
- how the data are scaled
- whether there is missing data and how it is coded
- whether there are outliers
- the distributions of continuous features
- the relationships between pairs of features

# **Data types**

It is important to check the data type for each feature. The quantitative variables should be read in as numbers — either **`int64`** or **`float64`** — and categorical variables should be stored as strings (columns of strings have a **`dtype`** of **`object`** because of how they are stored in Python). We can check data types of columns in a Pandas dataframe using **`.dtypes`**.

```
bb.dtypes

```

Output:

```
attendance     float64
game_type       object
day_of_week     object
temperature     object
sky             object
total_runs      object
dtype: object

```

From this output, we can see that **`temperature`** and **`total_runs`** are both quantitative variables but were read in as **`object`** dtypes. This can happen when there is a non-numeric character — such as a letter or punctuation symbol — in the same column. We would need to explore further in order to figure out what’s going on. For example, we might inspect a different set of rows and see the following:

[Untitled Database](https://www.notion.so/04be748d367a4d0db6ce06458e069e1a?pvs=21)

We note that **`temperature`** has missing data coded as **`Unknown`** rather than **`NaN`**. If we fit a regression on this data as is, we will end up treating temperature as a categorical variable and therefore fitting separate slopes for every value of temperature; instead, we probably want a single slope. To fix this, we’ll need to replace every **`Unknown`** with some other value (or remove them from the data altogether) and recode the temperature column as an **`int`**.

# **Categorical encoding**

EDA is also important during the feature engineering process in order to inform decisions around categorical encoding. This is important because categorical features with many levels are “expensive” to include in a regression model (we need to calculate a separate slope for each level). If one of the levels has only a few observations, we might want to delete those records from the data before fitting the model. We can check this using **`.value_counts()`**:

```
bb['game_type'].value_counts(dropna=False)

```

Output:

```
Night Game    1664
Day Game       799
Name: game_type, dtype: int64

```

Based on the output, we can see here that there are two levels for **`game_type`**; about one-third of games are day games and two-thirds are night games.

The **`.value_counts()`** accessor can also illuminate other issues. For example, in the following output, we notice that one instance of **`'Tuesday'`** was miscoded as **`Tuesda`**. This can either be corrected or removed before proceeding with a regression model.

```
bb['day_of_week'].value_counts(dropna=False)

```

Output:

```
Saturday     396
Friday       394
Sunday       392
Wednesday    379
Tuesday      375
Monday       278
Thursday     248
Tuesda         1
Name: day_of_week, dtype: int64

```

There are a few different options for how we might want to code the **`day_of_week`** variable. If attendance increases approximately linearly throughout the week, we might argue that **`day_of_week`** is ordinal and code it as an **`int`** in our model. However, attendance goes up and down throughout the week, we’re better off leaving it as an unordered category (**`str`**). Finally, if we see that games on Friday-Sunday simply have higher attendance than other days of the week, we might re-code this feature to only have two levels: **`Weekend`** and **`Weekday`**. We can check this by using boxplots:

![https://static-assets.codecademy.com/Courses/EDA/day_of_week_order_boxplot.svg](https://static-assets.codecademy.com/Courses/EDA/day_of_week_order_boxplot.svg)

We can see here that attendance on Friday, Saturday, and Sunday is on average higher than the other days of the week. Therefore it may be beneficial to re-code this feature to either **`Weekend`** or **`Weekday`**.

# **Scaling**

For quantitative features, it is important to think about how each feature is scaled. Some features will be on vastly different scales than others just based on the nature of what the feature is measuring. For example, let’s look at **`temperature`** and **`total_runs`** using the **`.describe()`** method:

```
bb.describe()

```

```
         attendance      temperature    total_runs
count    2457.000000     2457.000000    2457.000000
mean     30380.462352    73.834959      8.949187
std      9874.626652     10.567219      4.579542
min      8766.000000     31.000000      1.000000
25%      22437.000000    67.000000      6.000000
50%      30628.000000    74.000000      8.000000
75%      38412.000000    81.000000      12.000000
max      54449.000000    101.000000     60.000000

```

These two features are on different scales because what they are measuring are different (**`temperature`** is in degrees Fahrenheit, **`total_runs`** is the number of runs scored in a game). Because of this, the ranges of values and the standard deviations for each are very different from one another. We can see here that **`temperature`** has a standard deviation of about 10.57, while **`total_runs`** has a standard deviation of about 4.58.

When working with features with largely differing scales, it is often a good idea to *standardize* the features so that they all have a mean of 0 and a standard deviation of 1.

A feature without any values close to zero may also make it more difficult to estimate and interpret the intercept of a regression model. Standardizing or otherwise re-scaling the feature can fix this issue.

# **Missing data**

When we initially inspected the data, we saw some evidence that missing data is coded in a few different ways:

[Untitled Database](https://www.notion.so/89717544c10f43c48cc08c5a95b8c6c9?pvs=21)

For example, **`temperature`** uses the term **`Unknown`**, **`sky`** uses both **`Unknown`** and **`NaN`**, and **`total_runs`** has **`-`** to represent a missing value. The observations with missing values will either have to be removed or replaced (with an imputed value or missing data type that Python can recognize, such as **`np.NaN`**) in order to proceed with fitting a regression model.

# **Outliers**

In our EDA, it is important to check for outliers and skew in the data. One way to check for outliers is to use scatter plots:

```
bb.plot.scatter(x = 'total_runs',y = 'attendance')

```

![https://static-assets.codecademy.com/Courses/EDA/outlier.svg](https://static-assets.codecademy.com/Courses/EDA/outlier.svg)

We can see here that there is one instance where the total runs in a single game is about 60, which is much larger than in the other games. Depending on the situation, we may first want to verify that this value is correct, then we can decide whether or not to remove it prior to fitting the model.

# **Distributions and associations**

Prior to fitting a linear regression model, it can be important to inspect the distributions of quantitative features and investigate the relationships between features. We can visually inspect both of these by using a pair plot:

![https://static-assets.codecademy.com/Courses/EDA/pairplot.svg](https://static-assets.codecademy.com/Courses/EDA/pairplot.svg)

Looking at the histograms along the diagonal, **`total_runs`** appears to be somewhat right-skewed. This indicates that we may want to transform this feature to make it more normally distributed.

We can explore the relationships between pairs of features by looking at the scatterplots off of the diagonal. This is useful for a few different reasons. For example, if we see non-linear associations between any of the predictors and the outcome variable, that might lead us to test out polynomial terms in our model. We can also get a sense for which features are most highly related to our outcome variable and check for collinearity. In this example, there appears to be a slight positive linear association between temperature and the total number of runs. We can further investigate this using a heat map of the correlation matrix:

![https://static-assets.codecademy.com/Courses/EDA/heat_map.svg](https://static-assets.codecademy.com/Courses/EDA/heat_map.svg)

There is a correlation of 0.35 between temperature and the total number of runs. This is not large enough to cause concern; however, if two or more predictors are highly correlated, we may consider leaving only one in our analysis. On the other hand, features that are highly correlated with our outcome variable are especially important to include in the model.

# **Conclusion**

Let’s review the ways we were able to explore this data set in preparation for a regression model:

- We previewed the first few rows of the data set using the **`.head()`** method.
- We checked the data type of each variable in the data set using **`.dtypes`** and corrected variables with incorrect data types.
- We investigated our categorical data to inform categorical encoding
- We investigated the scale of our quantitative variables and considered whether standardizing/scaling might be appropriate
- We investigated missing data
- We checked for outliers
- We inspected the distributions of our quantitative variables
- We looked at the relationships between pairs of features using both scatter plots and box plots

By going through these steps, we are more prepared to make decisions about feature selection/engineering and have learned valuable information about how to build a more accurate predictive model.