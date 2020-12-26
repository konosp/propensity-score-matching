# Introduction
This notebook is used as a demonstration/introduction to propensity score matching. It uses the Kaggle Titanic dataset (https://www.kaggle.com/c/titanic). The main goal is to estimate the effect of cabin (i.e. treatment) on the final survival of passengers. 

The dataset helps illustrate the point that we cannot perform a RCT (randomised controlled testing) on the subjects. 
Also the results demonstrate a proof of concept but if more observations were available, the results would be more robust.

1. <a href='#Key-points'>Key points</a>
2. <a href='#Data-Preparation'>Data Preparation</a>
2. <a href='#Matching-Implementation'>Matching implementation</a>
3. <a href='#Matching-Review'>Matching Review</a>
4. <a href='#Average-Treatement-effect'>Average Treatement Effect</a>
1. <a href='#References'>References</a>

# Key points
In order to proceed to PSM (propensity score matching), the following key points are considered:
- Dimensions:
    - X are the underlying characteristics/features available.
    - T is the treatment; can be either 1 or 0. In this notebook the presence of a cabin is considered as T=1 (i.e. the passenger got treated).
    - Y is the outcome variable i.e. survived or not.
- Propensity score is the estimated probability that a subject/passenger is treated given certain observable characteristics X. In probability notation this is P(T=1|X). Propensity Score is used to "minimize" the dimensions. This solves the curse of dimensionality but on the other hand there is loss of information.
- The propensity score is calculated (usually) by logistic regression having T (treatment) as the outcome variable.
- There is a cost in not doing a proper RCT (randomised controlled testing). Treatment groups might not fully overlap (common support) or not all of characteristics X (i.e. age, fare etc.) might be equally distributed within the treatment groups.

Key assumptions & conditions:
- Identification condition: observations with similar characteristics X present in both treatment and control groups. This requires 0 < P(T=1|X) < 1
- Conditional independence assumption: There are no unobserved differences correlated to potential outcomes once we have controlled for certain observable characteristics
- Unconfoundedness assumption: Selection on treatment (or not) should be solely based on observable characteristics (i.e. X). Assuming there is no selection bias from unobserved characteristics. It is not possible to prove the validity of this unconfoundedness assumption.

# Data Preparation


```python
from sklearn.linear_model import LogisticRegression as lr

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
```


```python
from functions import *
import math
import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt
plt.style.use('classic')
%matplotlib inline

import seaborn as sns
sns.set(rc={'figure.figsize':(16,10)})
```

    
    Bad key "text.kerning_factor" on line 4 in
    /Users/konos/opt/anaconda3/lib/python3.7/site-packages/matplotlib/mpl-data/stylelib/_classic_test_patch.mplstyle.
    You probably need to get an updated matplotlibrc file from
    https://github.com/matplotlib/matplotlib/blob/v3.1.3/matplotlibrc.template
    or from the matplotlib source distribution



```python
df = pd.read_csv('train.csv')
# Elements are dropped for simplicity.
df = df[~df.Age.isna()]
df = df[~df.Embarked.isna()]
y = df[['Survived']]
df = df.drop(columns = ['Survived'])
```

Create an artificial treatment effect. It is based on the condition that a passenger has a cabin (1) or not (0).


```python
df['treatment'] = df.Cabin.apply(hasCabin)
```

There is high correlation between treatment (i.e. hasCabin) and Class.
This is desirable in this case as it plays the role of the systematic factor affecting the treatment.
In a different context this could be a landing page on site that only specific visitors see.


```python
pd.pivot_table(df[['treatment','Pclass','PassengerId']], \
               values = 'PassengerId', index = 'treatment', columns = 'Pclass',\
               aggfunc= np.count_nonzero)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Pclass</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
    <tr>
      <th>treatment</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>26</td>
      <td>158</td>
      <td>345</td>
    </tr>
    <tr>
      <th>1</th>
      <td>158</td>
      <td>15</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



Keeping only specific variables.
We should account for all variables that affect the treatment variable (i.e. hasCabin).


```python
df_data = df[['treatment','Sex','Age','SibSp','Parch','Embarked', 'Pclass', 'Fare']]
```


```python
T = df_data.treatment
X = df_data.loc[:,df_data.columns !='treatment']
```


```python
X_encoded = pd.get_dummies(X, columns = ['Sex','Embarked', 'Pclass'], \
                           prefix = {'Sex':'sex', 'Embarked' : 'embarked', 'Pclass' : 'class'}, drop_first=False)
```


```python
# Design pipeline to build the treatment estimator
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic_classifier', lr())
])

pipe.fit(X_encoded, T)
```




    Pipeline(steps=[('scaler', StandardScaler()),
                    ('logistic_classifier', LogisticRegression())])




```python
predictions = pipe.predict_proba(X_encoded)
predictions_binary = pipe.predict(X_encoded)
```


```python
print('Accuracy: {:.4f}\n'.format(metrics.accuracy_score(T, predictions_binary)))
print('Confusion matrix:\n{}\n'.format(metrics.confusion_matrix(T, predictions_binary)))
print('F1 score is: {:.4f}'.format(metrics.f1_score(T, predictions_binary)))
```

    Accuracy: 0.9284
    
    Confusion matrix:
    [[503  26]
     [ 25 158]]
    
    F1 score is: 0.8610


Convert propability to logit (based on the suggestion at https://youtu.be/gaUgW7NWai8?t=981)


```python
predictions_logit = np.array([logit(xi) for xi in predictions[:,1]])
```


```python
# Density distribution of propensity score (logic) broken down by treatment status
fig, ax = plt.subplots(1,2)
fig.suptitle('Density distribution plots for propensity score and logit(propensity score).')
sns.kdeplot(x = predictions[:,1], hue = T , ax = ax[0])
ax[0].set_title('Propensity Score')
sns.kdeplot(x = predictions_logit, hue = T , ax = ax[1])
ax[1].axvline(-0.4, ls='--')
ax[1].set_title('Logit of Propensity Score')
plt.show()
```


![png](images/output_19_0.png)


The graph on the right (logit_propensity_score) demonstrates the density for each treatment status. There is overlap accross the range of values (-6,5). However on the left of "-0.4" there are a lot more 0's than 1's. On the right side of "-0.4", the opposite is true (a lot more 1's than 0's). This will affect later how we will perform the matching so we can have balanced groups.


```python
# Currently this does not affect the results as all observations fall within this range.
common_support = (predictions_logit > -10) & (predictions_logit < 10)
```


```python
df_data.loc[:,'propensity_score'] = predictions[:,1]
df_data.loc[:,'propensity_score_logit'] = predictions_logit
df_data.loc[:,'outcome'] = y.Survived

X_encoded.loc[:,'propensity_score'] = predictions[:,1]
X_encoded.loc[:,'propensity_score_logit'] = predictions_logit
X_encoded.loc[:,'outcome'] = y.Survived
X_encoded.loc[:,'treatment'] = df_data.treatment
```

    /Users/konos/opt/anaconda3/lib/python3.7/site-packages/pandas/core/indexing.py:845: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.obj[key] = _infer_fill_value(value)
    /Users/konos/opt/anaconda3/lib/python3.7/site-packages/pandas/core/indexing.py:966: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.obj[item] = s



```python
X_encoded.loc[0]
```




    Age                       22.000000
    SibSp                      1.000000
    Parch                      0.000000
    Fare                       7.250000
    sex_female                 0.000000
    sex_male                   1.000000
    embarked_C                 0.000000
    embarked_Q                 0.000000
    embarked_S                 1.000000
    class_1                    0.000000
    class_2                    0.000000
    class_3                    1.000000
    propensity_score           0.021156
    propensity_score_logit    -3.834463
    outcome                    0.000000
    treatment                  0.000000
    Name: 0, dtype: float64




```python
df_data.treatment.value_counts()
```




    0    529
    1    183
    Name: treatment, dtype: int64



## Matching Implementation
Use Nearerst Neighbors to identify matching candidates. Then perform 1-to-1 matching by isolating/identifying groups of (T=1,T=0).
- Caliper: 25% of standart deviation of logit(propensity score)


```python
caliper = np.std(df_data.propensity_score) * 0.25

print('\nCaliper (radius) is: {:.4f}\n'.format(caliper))

df_data = X_encoded.loc[common_support].reset_index().rename(columns = {'index':'old_index'})

knn = NearestNeighbors(n_neighbors=10 , p = 2, radius=caliper)
knn.fit(df_data[['propensity_score_logit']].to_numpy())
```

    
    Caliper (radius) is: 0.0889
    





    NearestNeighbors(n_neighbors=10, radius=0.08890268148266278)



For each data point (based on the logit propensity score) obtain (at most) 10 nearest matches. This is regardless of their treatment status.


```python
# Common support distances and indexes
distances , indexes = knn.kneighbors(
    df_data[['propensity_score_logit']].to_numpy(), \
    n_neighbors=10)
```


```python
print('For item 0, the 4 closest distances are (first item is self):')
for ds in distances[0,0:4]:
    print('Element distance: {:4f}'.format(ds))
print('...')
```

    For item 0, the 4 closest distances are (first item is self):
    Element distance: 0.000000
    Element distance: 0.000021
    Element distance: 0.001106
    Element distance: 0.004490
    ...



```python
print('For item 0, the 4 closest indexes are (first item is self):')
for idx in indexes[0,0:4]:
    print('Element index: {}'.format(idx))
print('...')
```

    For item 0, the 4 closest indexes are (first item is self):
    Element index: 0
    Element index: 607
    Element index: 539
    Element index: 301
    ...


The overall space is split in two cases:
1. propensity_score_logit is greater than 0.4 - Less items with T=1 are present than T=0. This is used as the starting pool and select the closest item with T=0.
2. propensity_score_logit is less than 0.4 - Less items with T=0 are present than T=1. This is used as the starting pool and select the closest item with T=1.


```python
def perfom_matching(row, indexes, df_data):
    current_index = int(row['index']) # Obtain value from index-named column, not the actual DF index.
    prop_score_logit = row['propensity_score_logit']
    for idx in indexes[current_index,:]:
        if (prop_score_logit < -0.4) and (current_index != idx) and (row.treatment == 1) and (df_data.loc[idx].treatment == 0):
            return int(idx)
        elif (prop_score_logit > -0.4) and (current_index != idx) and (row.treatment == 0) and (df_data.loc[idx].treatment == 1):
            return int(idx)
        
df_data['matched_element'] = df_data.reset_index().apply(perfom_matching, axis = 1, args = (indexes, df_data))
```


```python
treated_with_match = ~df_data.matched_element.isna()
untreated_with_match = df_data.reset_index()['index'].isin(df_data.matched_element)
```


```python
all_matched_elements = pd.DataFrame(data = {'a' : treated_with_match, 'b' :untreated_with_match}).any(axis = 1)
```


```python
matched_data = df_data.loc[all_matched_elements]
```


```python
matched_data[['propensity_score','propensity_score_logit','outcome','treatment','matched_element']].head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>propensity_score</th>
      <th>propensity_score_logit</th>
      <th>outcome</th>
      <th>treatment</th>
      <th>matched_element</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>0.895107</td>
      <td>2.144005</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.805110</td>
      <td>1.418544</td>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.063216</td>
      <td>-2.695902</td>
      <td>1</td>
      <td>1</td>
      <td>686.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.059719</td>
      <td>-2.756534</td>
      <td>1</td>
      <td>1</td>
      <td>636.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.102161</td>
      <td>-2.173435</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Items that have matched_element = NaN, do not hold the matched-element information. This is held at their counterpart elements.


```python
matched_data.treatment.value_counts()
```




    0    51
    1    50
    Name: treatment, dtype: int64



In total 50 treated elements (i.e. hasCabin = True) have been matched with 51 untreated elements. A treated element has been matched twice.

# Matching Review


```python
fig, ax = plt.subplots(1,2)
fig.suptitle('Comparison of propensity scores split by treatment status.')
sns.kdeplot(data = df_data, x = 'propensity_score_logit', hue = 'treatment', ax = ax[0]).set(title='Distribution before matching')
sns.kdeplot(data = matched_data, x = 'propensity_score_logit', hue = 'treatment',  ax = ax[1]).set(title='Distribution after matching')
plt.show()
```


![png](images/output_41_0.png)



```python
def obtain_matched_pairs(row):
    x1 = row.Age
    y1 = row.Fare
   
    x2 = matched_data.loc[row.matched_element].Age
    y2 = matched_data.loc[row.matched_element].Fare
    return (x1, y1, x2, y2)
    
points = matched_data[~matched_data.matched_element.isna()].apply(obtain_matched_pairs, axis = 1)
```


```python
markers = {0: 'o', 1: '^'}
sns.scatterplot(data = matched_data, x = 'Age', y = 'Fare', style = 'treatment', s = 100, markers = markers, hue = 'propensity_score_logit').\
    set(xlim=(0, 75), ylim=(-10, 300))
plt.title('Matched pairs demonstration across Age and Fare')
for (x1, y1, x2, y2) in points:
     plt.plot([x1, x2], [y1, y2], linewidth=0.3, color = 'b', linestyle='dashed' )
```


![png](images/output_43_0.png)


The above chart demonstrates the result of the matching. Let's unpack the information it contains.
- Circles demonstrate untreated samples and Triangles treated samples.
- A circle is always matched with a triangle.
- Matched elements have similar propensity score (i.e. same color of shape).
- Matched elements might have big difference in some of their dimensions (i.e. Age gap) but their propensity score is always very close. This also demonstrates the fact that propensity scores lead to loss of information (due to the compression of multiple dimensions to a single number). 


```python
data = []
cols = ['Age','SibSp','Parch','Fare','sex_female','sex_male','embarked_C','embarked_Q','embarked_S','class_1','class_2','class_3']
# cols = ['Age','SibSp','Parch','Fare','sex_female','sex_male','embarked_C','embarked_Q','embarked_S']
for cl in cols:
    data.append([cl,'before', cohenD(df_data,cl)])
    data.append([cl,'after', cohenD(matched_data,cl)])
```


```python
res = pd.DataFrame(data, columns=['variable','matching','effect_size'])
```


```python
sns.barplot(data = res, y = 'variable', x = 'effect_size', hue = 'matching', orient='h').\
    set(title='Mean differences between treatment groups before and after matching.')
```




    [Text(0.5, 1.0, 'Mean differences between treatment groups before and after matching.')]




![png](images/output_47_1.png)



```python
cols.append('treatment')
```


```python
sns.pairplot(data = df_data[cols], hue = 'treatment')
print('Dimensions overview before matching')
```

    Dimensions overview before matching



![png](images/output_49_1.png)



```python
sns.pairplot(data = matched_data[cols], hue = 'treatment')
print('Dimensions overview after matching')
```

    /Users/konos/opt/anaconda3/lib/python3.7/site-packages/seaborn/distributions.py:305: UserWarning: Dataset has 0 variance; skipping density estimate.
      warnings.warn(msg, UserWarning)


    Dimensions overview after matching



![png](images/output_50_2.png)


# Average Treatement effect


```python
elements_a = matched_data[~matched_data.matched_element.isna()][['outcome','treatment','matched_element']]
elements_b = matched_data[matched_data.matched_element.isna()][['outcome','treatment']]
```


```python
combined_elements = pd.merge(left= elements_a, right= elements_b, \
                             left_on='matched_element', suffixes=('','_counterfactual'),  right_index=True)
```


```python
combined_elements['effect'] = 0
```


```python
# How to calculate treatement effect: https://youtu.be/CEikQRj5n_A?t=1044
def calc_effect(row):
    if (row.treatment == 1):
        return row.outcome - row.outcome_counterfactual
    elif (row.treatment == 0):
        return row.outcome_counterfactual - row.outcome
    
combined_elements['effect'] = combined_elements.apply(calc_effect, axis = 1)
```


```python
print('Average Treatement Effect (ATE): {:.4f}'.format(combined_elements['effect'].mean()))
print('Average Treatement Effect of Treated (ATT): {:.4f}'.format(combined_elements[combined_elements.treatment == 1]['effect'].mean()))
```

    Average Treatement Effect (ATE): 0.1569
    Average Treatement Effect of Treated (ATT): 0.3200



```python
print('Mean survival by treatment status:\n')
combined_elements[['treatment','outcome','outcome_counterfactual']].groupby(by = ['treatment']).aggregate(np.mean)
```

    Mean survival by treatment status:
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>outcome</th>
      <th>outcome_counterfactual</th>
    </tr>
    <tr>
      <th>treatment</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.538462</td>
      <td>0.538462</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.680000</td>
      <td>0.360000</td>
    </tr>
  </tbody>
</table>
</div>




```python
stats_results = stats.ttest_ind(combined_elements[combined_elements.treatment == 1]['effect'], combined_elements[combined_elements.treatment == 0]['effect'])
```


```python
print('p-value: {:.4f}'.format(stats_results.pvalue))
```

    p-value: 0.0758


# References
- The central role of the propensity score in observational studies for causal effects. https://academic.oup.com/biomet/article/70/1/41/240879?login=true
- Average Causal Effects From Nonrandomized Studies: A Practical Guide and Simulated Example https://www.researchgate.net/profile/Joseph_Kang4/publication/23652902_Average_Causal_Effects_From_Nonrandomized_Studies_A_Practical_Guide_and_Simulated_Example/links/565effb308ae1ef929843ae4/Average-Causal-Effects-From-Nonrandomized-Studies-A-Practical-Guide-and-Simulated-Example.pdf
- Logit function definition: https://en.wikipedia.org/wiki/Logit
- Matching Methods: https://www.youtube.com/watch?v=CEikQRj5n_A
- Propensity Score Matching: https://www.youtube.com/watch?v=8KE0qj5Ef0c
- Propensity Score Matching - HelloFresh: https://www.youtube.com/watch?v=gaUgW7NWai8
- Introduction to Propensity Score Matching: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483/
- https://florianwilhelm.info/2017/04/causal_inference_propensity_score/
- Cohen's d: https://en.wikipedia.org/wiki/Effect_size#Cohen's_d
- Python package pymatch: https://github.com/benmiroglio/pymatch
