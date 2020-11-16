# kddRadiomics
 A Radiomics Feature Extraction, Preprocessing, Feature Selecting, and Training Pipeline for Medical Images.

# Radiomics Feature Extraction

- `cfg.yaml` is global configuration file.

- `radiomics.yaml` is the configuration file for `PyRadiomics` library.

# Dataset Preprocessing

- `./preprocessing/dataset.py` is used for fetching features of training and testing dataset.

- `./preprocessing/featurePrepro.py` is used for pre-processing like imputation of null value and normalization.

# Feature Selecting

`./preprocessing/featureSelector.py`

## Chi-square test

The [chi-square test](https://en.wikipedia.org/wiki/Chi-squared_test) measures dependence between stochastic variables, so using this function “weeds out” the features that are the most likely to be independent of class and therefore irrelevant for classification.

##  ANOVA F-value

An [F-test](https://en.wikipedia.org/wiki/F-test) is any statistical test in which the test statistic has an F-distribution under the null hypothesis. It is most often used when comparing statistical models that have been fitted to a data set, in order to identify the model that best fits the population from which the data were sampled. 

## Mutual information

Estimate mutual information for a discrete target variable.

[Mutual information](https://en.wikipedia.org/wiki/Mutual_information) (MI) between two random variables is a non-negative value, which measures the dependency between the variables. More specifically, it quantifies the "amount of information" (in units such as shannons, commonly called bits) obtained about one random variable through observing the other random variable. It is equal to zero if and only if two random variables are independent, and higher values mean higher dependency.

## Linear regression

LinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.

## Lasso

[Lasso](https://en.wikipedia.org/wiki/Lasso_(statistics)) (least absolute shrinkage and selection operator; also Lasso or LASSO) is a regression analysis method that performs both variable selection and regularization in order to enhance the prediction accuracy and interpretability of the statistical model it produces. 

## Random forest

A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

## Correlation

- Pearson's correlation coefficient is the covariance of the two variables divided by the product of their standard deviations. 

- Kendall rank correlation coefficient is a statistic used to measure the ordinal association between two measured quantities.

- Spearman correlation coefficient is defined as the Pearson correlation coefficient between the rank variables.

## T test

A [t-test](https://www.investopedia.com/terms/t/t-test.asp) is a type of inferential statistic used to determine if there is a significant difference between the means of two groups, which may be related in certain features. It is mostly used when the data sets, like the data set recorded as the outcome from flipping a coin 100 times, would follow a normal distribution and may have unknown variances. A t-test is used as a hypothesis testing tool, which allows testing of an assumption applicable to a population. 

## Mann-Whitney U test

The [Mann-Whitney U test](https://statistics.laerd.com/spss-tutorials/mann-whitney-u-test-using-spss-statistics.php) is used to compare differences between two independent groups when the dependent variable is either ordinal or continuous, but not normally distributed. For example, you could use the Mann-Whitney U test to understand whether attitudes towards pay discrimination, where attitudes are measured on an ordinal scale, differ based on gender (i.e., your dependent variable would be "attitudes towards pay discrimination" and your independent variable would be "gender", which has two groups: "male" and "female"). 

# Training Pipeline

`./model/taskHelper.py`

- Classification

- Regression






