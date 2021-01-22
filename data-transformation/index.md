# Basic Feature Engineering Techniques for Machine Learning (under data heterogeneity)

In this topic I want to describe some besic data transformation and analysing techniques to prepare data for modeling.
For demontstration I took Medical Cost Personal Datasets from Kaggle.

Medical Cost Personal Datasets
Insurance Forecast by using Linear Regression
https://www.kaggle.com/mirichoi0218/insurance


```python
import pandas as pd
from matplotlib import pyplot as plt
import os
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy.stats import  ttest_ind
import scipy.stats as stats 
import warnings
warnings.filterwarnings('ignore')
```


```python
os.chdir('/python/insurence/')
```


```python
df = pd.read_csv("insurance.csv")
```


```python
df.head(5)
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
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>children</th>
      <th>smoker</th>
      <th>region</th>
      <th>charges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>female</td>
      <td>27.900</td>
      <td>0</td>
      <td>yes</td>
      <td>southwest</td>
      <td>16884.92400</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18</td>
      <td>male</td>
      <td>33.770</td>
      <td>1</td>
      <td>no</td>
      <td>southeast</td>
      <td>1725.55230</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>male</td>
      <td>33.000</td>
      <td>3</td>
      <td>no</td>
      <td>southeast</td>
      <td>4449.46200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>male</td>
      <td>22.705</td>
      <td>0</td>
      <td>no</td>
      <td>northwest</td>
      <td>21984.47061</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32</td>
      <td>male</td>
      <td>28.880</td>
      <td>0</td>
      <td>no</td>
      <td>northwest</td>
      <td>3866.85520</td>
    </tr>
  </tbody>
</table>
</div>



#### Numerical:

age: age of primary beneficiary

bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height,
objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9

charges: Individual medical costs billed by health insurance

children: Number of children covered by health insurance / Number of dependents

#### Categorical:

sex: insurance contractor gender, female, male

smoker: smoking or not

region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.


```python
df.describe()
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
      <th>age</th>
      <th>bmi</th>
      <th>children</th>
      <th>charges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1338.000000</td>
      <td>1338.000000</td>
      <td>1338.000000</td>
      <td>1338.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>39.207025</td>
      <td>30.663397</td>
      <td>1.094918</td>
      <td>13270.422265</td>
    </tr>
    <tr>
      <th>std</th>
      <td>14.049960</td>
      <td>6.098187</td>
      <td>1.205493</td>
      <td>12110.011237</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>15.960000</td>
      <td>0.000000</td>
      <td>1121.873900</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>27.000000</td>
      <td>26.296250</td>
      <td>0.000000</td>
      <td>4740.287150</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>39.000000</td>
      <td>30.400000</td>
      <td>1.000000</td>
      <td>9382.033000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>51.000000</td>
      <td>34.693750</td>
      <td>2.000000</td>
      <td>16639.912515</td>
    </tr>
    <tr>
      <th>max</th>
      <td>64.000000</td>
      <td>53.130000</td>
      <td>5.000000</td>
      <td>63770.428010</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1338 entries, 0 to 1337
    Data columns (total 7 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   age       1338 non-null   int64  
     1   sex       1338 non-null   object 
     2   bmi       1338 non-null   float64
     3   children  1338 non-null   int64  
     4   smoker    1338 non-null   object 
     5   region    1338 non-null   object 
     6   charges   1338 non-null   float64
    dtypes: float64(2), int64(2), object(3)
    memory usage: 73.3+ KB


## Charges distributon

Let's research the distributions for numerical features of the set.
Start with Charges and analythe it's distribution.
Our goal to make destribution normal for each feature before starting using this data in modeling.


```python
#Charges distributon
sns.distplot(df['charges'])
```
    
![png](images/output_12_1.png)
    


We can se that Charges normal distribution is asymmetrical.
A task of Data Scientist during feature preparation to achive data uniformity
For this there are some techniques applying (see more: https://www.analyticsvidhya.com/blog/2020/10/7-feature-engineering-techniques-machine-learning/): or https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114#1c08
- Imputation
- Handling Outliers
- Binning
- Log Transform
- Feature Encoding 
- Grouping Operations
- Scaling


 We will apply some of them depends of what behaviour we will have in features samples.

### Outliers for charges

Lets check how much outliers in Charges distribution


```python
sns.boxplot(x=df['charges'])
```

![png](images/output_16_1.png)
    


There are a lot of outliers from the right side. 
Tere are some imputation approaches for that. I'll try to delete outliers or replace outliers by median value and compare results.


```python
#Set feature
feature = 'charges'
```

#### Imputation and Handling outliers

Here it is calculating limit lower and upper values for a sample. Values over of this diapason (lover limit;upper limit) are outliers


```python
#calculate limit lower and upper values for a sample
def boundary_values (feature):
    feature_q25,feature_q75=np.percentile(df[feature],25),np.percentile(df[feature],75)
    feature_IQR=feature_q75-feature_q25
    Threshold=feature_IQR*1.5 #interquartile range (IQR)
    feature_lower,feature_upper=feature_q25-Threshold,feature_q75+Threshold
    print("Lower limit of "+feature+" distribution: "+str(feature_lower))
    print("Upper limit of "+feature+" distribution: "+str(feature_upper))
    return feature_lower,feature_upper;
    

#create two new DF with transformed/deleted outliers
#1st - outliers changed on sample median 
#2nd - deleting outliers 
def manage_outliers(df,feature_lower,feature_upper):
    df_del = df.copy()
    df_median = df.copy()
    
    median = df_del.loc[(df_del[feature]<feature_upper)&(df_del[feature]>feature_lower), feature].median()

    df_del.loc[(df_del[feature]> feature_upper)] = np.nan
    df_del.loc[(df_del[feature]< feature_lower)] = np.nan

    df_del.fillna(median,inplace=True)
   
    df_median.loc[(df_median[feature]> charges_upper)] = np.nan
    df_median.loc[(df_median[feature]< charges_lower)] = np.nan
    df_median.dropna(subset = [feature], inplace=True)

    return df_del,df_median;
```


```python
#calculate limits
x,y = boundary_values(feature)

#samples with modified outliers
df_median, df_del= manage_outliers(df,x,y)
```

    Lower limit of charges distribution: -13109.1508975
    Upper limit of charges distribution: 34489.350562499996



```python
df_median['charges'].mean()
```




    9770.084335798923




```python
df_agg = pd.DataFrame({'df':[df['charges'].mean(), df['charges'].max(), df['charges'].min(),df['charges'].std(),df['charges'].count()]}, columns = ['df'],index = ['mean','max','min','std','count'])
df_agg['df_mean'] = pd.DataFrame({'df_median':[df_median['charges'].mean(), df_median['charges'].max(), df_median['charges'].min(),df_median['charges'].std(),df_median['charges'].count()]}, columns = ['df_median'],index = ['mean','max','min','std','count'])
df_agg['df_del'] = pd.DataFrame({'df_del':[df_del['charges'].mean(), df_del['charges'].max(), df_del['charges'].min(),df_del['charges'].std(),df_del['charges'].count()]}, columns = ['df_del'],index = ['mean','max','min','std','count'])
df_agg
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
      <th>df</th>
      <th>df_mean</th>
      <th>df_del</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean</th>
      <td>13270.422265</td>
      <td>9770.084336</td>
      <td>9927.753402</td>
    </tr>
    <tr>
      <th>max</th>
      <td>63770.428010</td>
      <td>34472.841000</td>
      <td>34472.841000</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1121.873900</td>
      <td>1121.873900</td>
      <td>1121.873900</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12110.011237</td>
      <td>6870.056585</td>
      <td>7241.158309</td>
    </tr>
    <tr>
      <th>count</th>
      <td>1338.000000</td>
      <td>1338.000000</td>
      <td>1199.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#the sample after sample modification

fig, ax = plt.subplots(2,2, figsize=(15, 5))

sns.distplot(ax = ax[0,0], x = df['charges'])
sns.distplot(ax = ax[1,0], x = df_median['charges'])
sns.distplot(ax = ax[1,1], x= df_del['charges'])
plt.show()
```


    
![png](images/output_25_0.png)
    



```python
df_shape = df.agg(['skew', 'kurtosis']).transpose()
df_shape.rename(columns = {'skew':'skew_df','kurtosis':'kurtosis_df'}, inplace = True)
df_shape['skew_median'] = df_median.agg(['skew', 'kurtosis']).transpose()['skew']
df_shape['kurtosis_median'] = df_median.agg(['skew', 'kurtosis']).transpose()['kurtosis']
df_shape['skew_del'] = df_del.agg(['skew', 'kurtosis']).transpose()['skew']
df_shape['kurtosis_del'] = df_del.agg(['skew', 'kurtosis']).transpose()['kurtosis']
df_shape
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
      <th>skew_df</th>
      <th>kurtosis_df</th>
      <th>skew_median</th>
      <th>kurtosis_median</th>
      <th>skew_del</th>
      <th>kurtosis_del</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>0.055673</td>
      <td>-1.245088</td>
      <td>2.599285</td>
      <td>4.763691</td>
      <td>0.067588</td>
      <td>-1.255101</td>
    </tr>
    <tr>
      <th>bmi</th>
      <td>0.284047</td>
      <td>-0.050732</td>
      <td>2.599394</td>
      <td>4.764021</td>
      <td>0.366750</td>
      <td>0.011529</td>
    </tr>
    <tr>
      <th>children</th>
      <td>0.938380</td>
      <td>0.202454</td>
      <td>2.599417</td>
      <td>4.764092</td>
      <td>0.987108</td>
      <td>0.318218</td>
    </tr>
    <tr>
      <th>charges</th>
      <td>1.515880</td>
      <td>1.606299</td>
      <td>1.304122</td>
      <td>1.565420</td>
      <td>1.178483</td>
      <td>1.022970</td>
    </tr>
  </tbody>
</table>
</div>



Look in charges row. Here is some positive changes: skew and kurtosis decreeased, but still not significantly. So charges feater still need additional improvements.

There are a few data transformation methods solwing abnormality. Let's try there two of them (Square Root and Log) and choose better for the data set.

#### Square Root transformation

 Square root method is typically used when your data is moderately skewed  (see more: https://www.marsja.se/transform-skewed-data-using-square-root-log-box-cox-methods-in-python/)


```python
df.insert(len(df.columns), 'charges_Sqrt',np.sqrt(df.iloc[:,6]))
df_median.insert(len(df_median.columns), 'charges_Sqrt',np.sqrt(df_median.iloc[:,6]))
df_del.insert(len(df_del.columns), 'charges_Sqrt',np.sqrt(df_del.iloc[:,6]))
```


```python
fig, ax = plt.subplots(2,2, figsize=(15, 5))

sns.distplot(ax = ax[0,0], x = df['charges_Sqrt'])
sns.distplot(ax = ax[1,0], x = df_median['charges_Sqrt'])
sns.distplot(ax = ax[1,1], x= df_del['charges_Sqrt'])
plt.show()
```


    
![png](images/output_32_0.png)
    



```python
df_shape_sqrt = df.agg(['skew', 'kurtosis']).transpose()
df_shape_sqrt.rename(columns = {'skew':'skew_df','kurtosis':'kurtosis_df'}, inplace = True)
df_shape_sqrt['skew_median'] = df_median.agg(['skew', 'kurtosis']).transpose()['skew']
df_shape_sqrt['kurtosis_median'] = df_median.agg(['skew', 'kurtosis']).transpose()['kurtosis']
df_shape_sqrt['skew_del'] = df_del.agg(['skew', 'kurtosis']).transpose()['skew']
df_shape_sqrt['kurtosis_del'] = df_del.agg(['skew', 'kurtosis']).transpose()['kurtosis']
df_shape_sqrt
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
      <th>skew_df</th>
      <th>kurtosis_df</th>
      <th>skew_median</th>
      <th>kurtosis_median</th>
      <th>skew_del</th>
      <th>kurtosis_del</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>0.055673</td>
      <td>-1.245088</td>
      <td>2.599285</td>
      <td>4.763691</td>
      <td>0.067588</td>
      <td>-1.255101</td>
    </tr>
    <tr>
      <th>bmi</th>
      <td>0.284047</td>
      <td>-0.050732</td>
      <td>2.599394</td>
      <td>4.764021</td>
      <td>0.366750</td>
      <td>0.011529</td>
    </tr>
    <tr>
      <th>children</th>
      <td>0.938380</td>
      <td>0.202454</td>
      <td>2.599417</td>
      <td>4.764092</td>
      <td>0.987108</td>
      <td>0.318218</td>
    </tr>
    <tr>
      <th>charges</th>
      <td>1.515880</td>
      <td>1.606299</td>
      <td>1.304122</td>
      <td>1.565420</td>
      <td>1.178483</td>
      <td>1.022970</td>
    </tr>
    <tr>
      <th>charges_Sqrt</th>
      <td>0.795863</td>
      <td>-0.073061</td>
      <td>0.479111</td>
      <td>-0.089718</td>
      <td>0.440043</td>
      <td>-0.399537</td>
    </tr>
  </tbody>
</table>
</div>



#### Log transformation


```python
# Python log transform
df.insert(len(df.columns), 'charges_log',np.log(df['charges']))
df_median.insert(len(df_median.columns), 'charges_log',np.log(df_median['charges']))
df_del.insert(len(df_del.columns), 'charges_log',np.log(df_del['charges']))
```


```python
df_shape_log = df.agg(['skew', 'kurtosis']).transpose()
df_shape_log.rename(columns = {'skew':'skew_df','kurtosis':'kurtosis_df'}, inplace = True)
df_shape_log['skew_median'] = df_median.agg(['skew', 'kurtosis']).transpose()['skew']
df_shape_log['kurtosis_median'] = df_median.agg(['skew', 'kurtosis']).transpose()['kurtosis']
df_shape_log['skew_del'] = df_del.agg(['skew', 'kurtosis']).transpose()['skew']
df_shape_log['kurtosis_del'] = df_del.agg(['skew', 'kurtosis']).transpose()['kurtosis']
df_shape_log
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
      <th>skew_df</th>
      <th>kurtosis_df</th>
      <th>skew_median</th>
      <th>kurtosis_median</th>
      <th>skew_del</th>
      <th>kurtosis_del</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>0.055673</td>
      <td>-1.245088</td>
      <td>2.599285</td>
      <td>4.763691</td>
      <td>0.067588</td>
      <td>-1.255101</td>
    </tr>
    <tr>
      <th>bmi</th>
      <td>0.284047</td>
      <td>-0.050732</td>
      <td>2.599394</td>
      <td>4.764021</td>
      <td>0.366750</td>
      <td>0.011529</td>
    </tr>
    <tr>
      <th>children</th>
      <td>0.938380</td>
      <td>0.202454</td>
      <td>2.599417</td>
      <td>4.764092</td>
      <td>0.987108</td>
      <td>0.318218</td>
    </tr>
    <tr>
      <th>charges</th>
      <td>1.515880</td>
      <td>1.606299</td>
      <td>1.304122</td>
      <td>1.565420</td>
      <td>1.178483</td>
      <td>1.022970</td>
    </tr>
    <tr>
      <th>charges_Sqrt</th>
      <td>0.795863</td>
      <td>-0.073061</td>
      <td>0.479111</td>
      <td>-0.089718</td>
      <td>0.440043</td>
      <td>-0.399537</td>
    </tr>
    <tr>
      <th>charges_log</th>
      <td>-0.090098</td>
      <td>-0.636667</td>
      <td>-0.393856</td>
      <td>-0.319703</td>
      <td>-0.328473</td>
      <td>-0.609327</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(2,2, figsize=(15, 5))

sns.distplot(ax = ax[0,0], x = df['charges_log'])
sns.distplot(ax = ax[1,0], x = df_median['charges_log'])
sns.distplot(ax = ax[1,1], x= df_del['charges_log'])
plt.show()
```


    
![png](images/output_37_0.png)
    


Here in table we can compare pairs skew-kurtosis for three DF: unmodified, with outliers changed on mean and with deleted outliers.
The first three pare for Charges DF looks non-normal, because both in pair are enough far from 0.
After Square-Root transformations the best pair is a mediand_df pair with lower skew and kurtosis in the same time.
If compare with Log-transformation, the best values for normal distribution is in the initial DF.
For df_del there are a good result in solwing skewness issue.
Log-transformation works good with assimetrical data.
IF compare shapes on the graphs, we see there that initial DF is more symmetrical.

#### Interim conclusions: Desribution is still non-normal. But anyway the previous transformations get us some enough good results and allow to work with data further. For a modeling it makes sense to use log-transformed charges or square-root-transformed charges and outliers replaced by medians. Deleting outliers help just partly with only with kurtosis issue.

#### Addition:

The Box-Cox transformation is a technique to transform non-normal data into normal shape.
Box-cox transformation attempts to transform a set of data to a normal distribution by finding the value of λ that minimizes the variation. (see more: https://medium.com/@ronakchhatbar/box-cox-transformation-cba8263c5206)


```python
skewed_box_cox, lmda = stats.boxcox(df['charges'])
sns.distplot(skewed_box_cox)
```


    
![png](images/output_42_0.png)
    



```python
df['boxcox'].skew()
```




    -0.008734097133920404




```python
df['boxcox'].kurtosis()
```




    -0.6502935539475279



#### Box-cox gives good results and can be used for 'charges' as Log-transformation

### BMI Distribution


```python
sns.distplot(df['bmi'])
```

![png](images/output_47_1.png)
    



```python
At first glance, the distribution looks normal.
```
#### Shapiro Normality test

There is one more test allows to check normality of destribution. It is Shapiro test. For this spicy library can be use


```python
from scipy import stats
p_value = stats.shapiro(df['bmi'])[0]
if p_value <=0.05:
    print("Distribution is non-normal")
else:
    print('Distribution is normal')
```

    Distribution is normal



```python
df.agg(['skew', 'kurtosis'])['bmi'].transpose()
```




    skew        0.284047
    kurtosis   -0.050732
    Name: bmi, dtype: float64



##### Interim conclusions: BMI index destributed normally


### Age Distributon 

As we see on the plot, ages density is quite equal, exept age near 20. Let's take a look deeper


```python
sns.distplot(df['age'])
```

![png](images/output_56_1.png)
    


#### Outliers for Age


```python
sns.boxplot(x=df['age'])
```


    
![png](images/output_58_1.png)
    



```python
df.agg(['skew', 'kurtosis'])['age'].transpose()
```




    skew        0.055673
    kurtosis   -1.245088
    Name: age, dtype: float64




```python
df.describe()['age']
```




    count    1338.000000
    mean       39.207025
    std        14.049960
    min        18.000000
    25%        27.000000
    50%        39.000000
    75%        51.000000
    max        64.000000
    Name: age, dtype: float64




```python
df.describe()
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
      <th>age</th>
      <th>bmi</th>
      <th>children</th>
      <th>charges</th>
      <th>charges_Sqrt</th>
      <th>charges_log</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1338.000000</td>
      <td>1338.000000</td>
      <td>1338.000000</td>
      <td>1338.000000</td>
      <td>1338.000000</td>
      <td>1338.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>39.207025</td>
      <td>30.663397</td>
      <td>1.094918</td>
      <td>13270.422265</td>
      <td>104.833605</td>
      <td>9.098659</td>
    </tr>
    <tr>
      <th>std</th>
      <td>14.049960</td>
      <td>6.098187</td>
      <td>1.205493</td>
      <td>12110.011237</td>
      <td>47.770734</td>
      <td>0.919527</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>15.960000</td>
      <td>0.000000</td>
      <td>1121.873900</td>
      <td>33.494386</td>
      <td>7.022756</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>27.000000</td>
      <td>26.296250</td>
      <td>0.000000</td>
      <td>4740.287150</td>
      <td>68.849739</td>
      <td>8.463853</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>39.000000</td>
      <td>30.400000</td>
      <td>1.000000</td>
      <td>9382.033000</td>
      <td>96.860893</td>
      <td>9.146552</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>51.000000</td>
      <td>34.693750</td>
      <td>2.000000</td>
      <td>16639.912515</td>
      <td>128.995729</td>
      <td>9.719558</td>
    </tr>
    <tr>
      <th>max</th>
      <td>64.000000</td>
      <td>53.130000</td>
      <td>5.000000</td>
      <td>63770.428010</td>
      <td>252.528074</td>
      <td>11.063045</td>
    </tr>
  </tbody>
</table>
</div>




```python
#samples with modified outliers
#calculate limits
feature = 'age'
x,y = boundary_values(feature)
```

    Lower limit of age distribution: -9.0
    Upper limit of age distribution: 87.0

    -9.0


So we see that there is no outliers in this distribution

Let's see "left side" counts by ages


```python
df.groupby(['age'])['age'].count().head(10)
```


    age
    18    69
    19    68
    20    29
    21    28
    22    28
    23    28
    24    28
    25    28
    26    28
    27    28
    Name: age, dtype: int64



As we see from the histogram and last output that it is near 2 times more data near 20 years, and it shouls be corrected.
I want to find median value count for age and decrease diapasone of 18-19 years old till this median.


```python
n = int(df['age'].value_counts().median())
```


```python
df1 = df.copy()

df_19 = df1[(df1['age']==19)]
df_18 = df1[(df1['age']==18)]
df_19.describe()
df_18.iloc[n:df2.size,:].index
df_19.iloc[n:df2.size,:].index
df = df.drop(df_18.iloc[n:df2.size,:].index)
df = df.drop(df_19.iloc[n:df2.size,:].index)

```


```python
df.describe()['age']
```




    count    1255.000000
    mean       40.576892
    std        13.422954
    min        18.000000
    25%        29.000000
    50%        41.000000
    75%        52.000000
    max        64.000000
    Name: age, dtype: float64




```python
sns.distplot(df['age'])
```
    
![png](images/output_70_1.png)
    



```python
df.agg(['skew', 'kurtosis'])['age'].transpose()
```


    skew        0.004377
    kurtosis   -1.195052
    Name: age, dtype: float64



So dropping reduced skewness and kurtosis a little bit.

##### Interim conclusions: It make sence there to leave this distribution as it is because it is performe all ages more-less equally and doesn't need to be more normal distributed.

### Conclusions: 
#### In this article I described the most typical, often used and effective transformation approaches to get normal distibution. This transformations are important for the further modeling applying. Some model are sensitive for the data view and data scientis have to investigate more in a data preparation.
#### As a result we can see, that Log-transformation is the most universal and effective one technique. It solve most of the skeweness and kurtosis problems. Box-Cox transformations are the same effective and flexible.
#### It can happend that data looks non-normal, but in the same time it doesn't have some outliers or very high kurtosis. In this situation it make sence analyse data locally and adjust it manually, for example deleting data or replacing it for a median/mean/max/min/random etc. values.

