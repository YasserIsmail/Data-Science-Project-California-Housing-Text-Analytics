
# 🧠 Data Analysis and Preprocessing Project

This project demonstrates the complete process of analyzing housing and text data using Python libraries such as Pandas, Seaborn, Matplotlib, GeoPandas, and Scikit-learn. It includes data cleaning, visualization, feature engineering, geospatial plotting, text preprocessing, and basic machine learning preparation.

---

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
```

## 1. Load the housing.csv file using pandas.


```python
df = pd.read_csv('housing.csv')
```

## 2. Drop all rows with at least one missing value.


```python
df.dropna(inplace=True)
```

## 3. rooms_per_household


```python
df['rooms_per_household'] = df['total_rooms']/df['households']
```

## 4. rooms_per_householdpopulation_per_household


```python
df['population_per_household'] = df['total_rooms']/df['population']
```

## 5. Calculate the average house value for houses that are 50 years old or less. 


```python
housing_median_age_lessEqual50= df.loc[(df['housing_median_age']<=50),'median_house_value']
housing_median_age_lessEqual50.mean()
```




    202288.8914363484



## 6. Calculate the correlation of “median_house_value”


```python
median_house_correlation = df.corr()[['median_house_value']]
```


```python
median_house_correlation
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
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>longitude</th>
      <td>-0.045463</td>
    </tr>
    <tr>
      <th>latitude</th>
      <td>-0.144550</td>
    </tr>
    <tr>
      <th>housing_median_age</th>
      <td>0.106439</td>
    </tr>
    <tr>
      <th>total_rooms</th>
      <td>0.133148</td>
    </tr>
    <tr>
      <th>total_bedrooms</th>
      <td>0.049581</td>
    </tr>
    <tr>
      <th>population</th>
      <td>-0.025416</td>
    </tr>
    <tr>
      <th>households</th>
      <td>0.064785</td>
    </tr>
    <tr>
      <th>median_income</th>
      <td>0.688308</td>
    </tr>
    <tr>
      <th>median_house_value</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>rooms_per_household</th>
      <td>0.151237</td>
    </tr>
    <tr>
      <th>population_per_household</th>
      <td>0.208461</td>
    </tr>
  </tbody>
</table>
</div>




```python
median_house_correlation.sort_values(by='median_house_value',inplace=True)
```


```python
plt.figure(figsize=(4, 8))
sns.heatmap(median_house_correlation, annot=True, cmap='rocket')
plt.title('\'median_house_value\' Correlation Matrix Heatmap',pad=20,fontsize=15)
```




    Text(0.5, 1.0, "'median_house_value' Correlation Matrix Heatmap")




    
![png](output_15_1.png)
    


### Which factors seem to influence house prices?

> The greatest feature is influences **'median_house_value'** is **'median_income'** which is related to as posisitive correlation followed it the **'population_per_households'** and the **'rooms_per_households'** with fewer positive corrlation.

>The other features don't have corrleatiions with **'median_house_value'**.

>**'latitude'** and **'longtitude'** are spatial data.

>I don't see a negative correlation.

## 7. Create a Seaborn regression plot (jointplot) with income in the a-axis and house value on the y-axis


```python
plt.figure(figsize=(12, 8))
sns.jointplot(x='median_income', y='median_house_value', data=df, kind='reg' ,line_kws={'color': 'red'}, height=8)
```




    <seaborn.axisgrid.JointGrid at 0x1c7eef76910>




    <Figure size 864x576 with 0 Axes>



    
![png](output_19_2.png)
    


## 8. Create the following scatterplot:
 - Longitude on x-axis
 - Latitude on y-axis
 - Size of datapoints is determined by population
 - Color of datapoints is determined by median_house_value

## 9. Use geo_pandas to draw the previous scatter plot on California map.


```python
gdf = gpd.read_file('ca_state.zip')
gdf.to_crs(4326,inplace=True)
```


```python
axis = gdf.plot(figsize=(10,12),color='lightgray', edgecolor='black', alpha=0.3)
sns.scatterplot(
    ax=axis,
    data=df,
    x='longitude',
    y='latitude',
    hue='median_house_value',
    size='population',
    palette='coolwarm',
    edgecolor='black',
    sizes=(20,250),
    alpha=0.6,
)
axis.set_title('Median House Value and Population', fontsize=16)
axis.set_xlabel('Longitude', fontsize=12)
axis.set_ylabel('Latitude', fontsize=12)
plt.legend(fontsize=12, bbox_to_anchor=(1, 1), loc='upper left')
```




    <matplotlib.legend.Legend at 0x1c7eefa66d0>




    
![png](output_23_1.png)
    


## 10.Add the additional column “income_cat” with the following income categories:


```python
q1   = df['median_income'].quantile(0.25)
q2   = df['median_income'].quantile(0.5)
q3   = df['median_income'].quantile(0.75)
at95 = df['median_income'].quantile(0.95)
```


```python
df['income_cat'] = pd.Series(dtype='object')
```


```python
df.loc[df['median_income']<q1,'income_cat'] = 'Low'
df.loc[(df['median_income']>=q1)&(df['median_income']<q2),'income_cat'] = 'Below average'
df.loc[(df['median_income']>=q2)&(df['median_income']<q3),'income_cat'] = 'Above average'
df.loc[(df['median_income']>=q3)&(df['median_income']<at95),'income_cat'] = 'High'
df.loc[df['median_income']>=at95,'income_cat'] = 'Very high'
```

## 11. Create the following plot using seaborn:


```python
plt.figure(figsize=(12, 8))
sns.histplot(data=df.sort_values(by='median_income'),x='income_cat',hue='ocean_proximity',multiple='dodge')
plt.grid(True, alpha=0.6,axis='y')
```


    
![png](output_29_0.png)
    


## 12. Create the following plot using Seaborn


```python
plt.figure(figsize=(12, 8))
plt.grid(True, alpha=0.6,axis='y')
sns.barplot(x='income_cat',y='median_house_value',data=df.sort_values(by='median_income'))
```




    <AxesSubplot:xlabel='income_cat', ylabel='median_house_value'>




    
![png](output_31_1.png)
    


## 13. Create the following plot using Seaborn


```python
plt.figure(figsize=(12, 8))
sns.barplot(x='ocean_proximity',y='median_house_value',data=df)
plt.grid(True, alpha=0.6,axis='y')
```


    
![png](output_33_0.png)
    


## 14. Create the following Seaborn Heatmap with mean house values for all combinations of income_cat and ocean_proximity.


```python
heatmap_df = df[['ocean_proximity','income_cat','median_house_value']]
```


```python
groups = heatmap_df.groupby(['ocean_proximity','income_cat'])
```


```python
df_list = []
for i in groups.groups:
    df_list.append([i[0],i[1],groups.get_group(i)['median_house_value'].mean()])
```


```python
heatmap_df = pd.DataFrame(df_list,columns=['ocean_proximity','income_cat','median_house_value'])
```


```python
heatmap_df = heatmap_df.pivot('income_cat', 'ocean_proximity','median_house_value')
```


```python
heatmap_df.drop(columns=['ISLAND'],inplace=True)
```


```python
heatmap_df.sort_values(by=['<1H OCEAN','INLAND','NEAR BAY','NEAR OCEAN'],inplace=True)
```


```python
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_df, annot=True, fmt=".0f", linewidths=0.5)
```




    <AxesSubplot:xlabel='ocean_proximity', ylabel='income_cat'>




    
![png](output_42_1.png)
    


## 15.Reload the housing.csv file again, and handle missing data as the following:


```python
df = pd.read_csv('housing.csv')
```

- Missing values in the total_bedrooms column should be filled with the mean value.


```python
mean = df['total_bedrooms'].mean()
df['total_bedrooms'] = df['total_bedrooms'].fillna(mean)
```


```python
#from sklearn.impute import SimpleImputer 
#imputer = SimpleImputer(strategy='mean')
#df['total_bedrooms'] = imputer.fit_transform(df[['total_bedrooms']])
```

- Missing values in the total_rooms column should be filled with the median value.


```python
median = df['total_rooms'].median()
df['total_rooms'] = df['total_rooms'].fillna(median)
```


```python
#from sklearn.impute import SimpleImputer 
#imputer = SimpleImputer(strategy='median')
#df['total_rooms'] = imputer.fit_transform(df[['total_rooms']])
```

- Missing values in the total_rooms column should be filled with the value calculated by KNN algorithm, with k=2.


```python
#from sklearn.impute import KNNImputer
#knn_imputer = KNNImputer(n_neighbors=2)
#num_df = df.select_dtypes(np.number)
#df_imputed = pd.DataFrame(knn_imputer.fit_transform(num_df),columns=num_df.columns)
#df['total_rooms'] = df_imputed['total_rooms']
```

## 16. Identify columns that have outliers by using the boxplot of Seaborn or Matplotlib.


```python
num_df = df.select_dtypes(np.number)
```


```python
plt.figure(figsize=(15, 8))
ncols = 3
num_features = len(num_df.columns)
nrows = (num_features + 1) // ncols
for i, column in enumerate(num_df.columns, 1):
    plt.subplot(nrows,ncols, i)
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
plt.tight_layout()
plt.show()
```


    
![png](output_55_0.png)
    


## 17. Write python function to remove outliers from all columns by using 1.5*IQR rule.


```python
def drop_outliers(df):
    columns = df.select_dtypes(np.number)
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3-q1
        lower_bound = q1-1.5*iqr
        upper_bound = q3+1.5*iqr
        df = df.loc[(df[col]>=lower_bound)&(df[col]<=upper_bound)]
    return df
```


```python
df_droped_outlier = drop_outliers(df)
```


```python
nrows = len(num_df.columns)
fig, axes = plt.subplots(nrows, 2, figsize=(15, 30))
for i, column in enumerate(num_df.columns):
    
    # Left column (first axis)
    sns.boxplot(x=df[column], ax=axes[i, 0]) 
    axes[i, 0].set_title(f'{column}',fontsize=14, fontweight='bold')
    
    # Right column (second axis)
    sns.boxplot(x=df_droped_outlier[column], ax=axes[i, 1])  
    axes[i, 1].set_title(f'{column}',fontsize=14, fontweight='bold')    
    
fig.text(0.25, 0.95, 'Left: Original Data', ha='center', va='center', fontsize=16, fontweight='bold')

fig.text(0.75, 0.95, 'Right: Data with Outliers Removed', ha='center', va='center', fontsize=16, fontweight='bold')

plt.tight_layout()
fig.subplots_adjust(top=0.92, hspace=0.5)
```


    
![png](output_59_0.png)
    


## 18. We want to develop a model that should predict the median house value given the other attributes. Separate the median_house_value so that it can be used as the labels columns. Other columns will be used as features for machine learning.


```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
```


```python
X = df.drop(columns=['longitude','latitude','median_house_value']) ##'latitude' and 'longtitude'** are spatial data.
y = df['median_house_value']
```

## 19. Split into train and test sets.


```python
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
```

## 20. Normalize numeric columns by using MinMaxScaler from sklearn.
## 21. Use OneHotEncoding to encode the ocean_proximity column. 
## 22. Use ColumnTransformer from sklearn to apply the previous twotransformation processes on the features.


```python
scaler = MinMaxScaler()
encoder = OneHotEncoder(handle_unknown='ignore')
```


```python
num = X.select_dtypes(np.number)
cat = X.select_dtypes(exclude=[np.number])
```


```python
processor = ColumnTransformer([
    ('scaler', scaler, num.columns),
    ('encoder', encoder, cat.columns)
])
```


```python
processed_X_train = processor.fit_transform(X_train)
processed_X_test = processor.transform(X_test)
```


```python
processed_y_train = scaler.fit_transform(np.array(y_train.to_numpy().reshape(-1,1)))
processed_y_test  = scaler.transform(y_test.to_numpy().reshape(-1,1))
```


##Yasser I Barhoom (Geomatics Eng.)
```


```python

```
