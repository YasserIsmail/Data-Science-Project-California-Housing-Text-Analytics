# 🏠 California Housing Data Analysis

This project provides a comprehensive analysis and visualization of the California housing dataset using Python, including:
- Data cleaning and feature engineering
- Statistical summaries and correlations
- Geospatial visualizations with GeoPandas
- Outlier detection and removal
- Machine learning preprocessing


### 📦 **Import Libraries**

```python
# Importing pandas for data manipulation and analysis (tables, DataFrames)
import pandas as pd

# Importing numpy for numerical computations and working with arrays
import numpy as np

# Importing seaborn for enhanced data visualization (built on matplotlib)
import seaborn as sns

# Importing matplotlib for plotting and visualizing data (line charts, histograms, etc.)
import matplotlib.pyplot as plt

# Importing geopandas for working with geospatial data (e.g., shapefiles, GeoJSON)
import geopandas as gpd
````

> These libraries form the core stack for data analysis, visualization, and geospatial processing in Python.
>
> * `pandas` and `numpy`: essential for handling structured and numerical data.
> * `seaborn` and `matplotlib`: used to create clear and insightful visualizations.
> * `geopandas`: enables spatial analysis using vector data like points, lines, and polygons.


### 📂 **2. Load the `housing.csv` file**

```python
# Load the housing dataset into a pandas DataFrame
df = pd.read_csv('housing.csv')
```

> This command reads the `housing.csv` file and stores it in the `df` variable as a DataFrame.  
> The DataFrame allows for structured analysis, filtering, and visualization of the housing data.


### 🧹 **2. Drop Missing Values**

```python
df.dropna(inplace=True)
```

> Remove all rows that contain any missing values.

---

### 🧮 **3. Compute rooms per household**

```python
df['rooms_per_household'] = df['total_rooms']/df['households']
```

> Add a new column that calculates average rooms per household.

---

### 🧮 **4. Compute population per household**

```python
df['population_per_household'] = df['total_rooms']/df['population']
```

> Add a new column that calculates rooms per person (likely meant to be population per household).

---

### 📊 **5. Average house value for homes <= 50 years**

```python
housing_median_age_lessEqual50= df.loc[(df['housing_median_age']<=50),'median_house_value']
housing_median_age_lessEqual50.mean()
```

> Filter and calculate mean house value for houses 50 years old or less.

---

### 🔗 **6. Calculate correlation with house value**

```python
median_house_correlation = df.corr()[['median_house_value']]
```

> Generate correlation of each column with `median_house_value`.

```python
median_house_correlation
```

> Display correlation table.

```python
median_house_correlation.sort_values(by='median_house_value',inplace=True)
```

> Sort the correlation values in ascending order.

```python
plt.figure(figsize=(4, 8))
sns.heatmap(median_house_correlation, annot=True, cmap='rocket')
plt.title('\'median_house_value\' Correlation Matrix Heatmap',pad=20,fontsize=15)
```

> Visualize the sorted correlation values using a heatmap.

---

### 🖊 **Comment on Correlations**

> Provide interpretation of what features influence house values.

---

### 📈 **7. Regression Plot: Income vs House Value**

```python
plt.figure(figsize=(12, 8))
sns.jointplot(x='median_income', y='median_house_value', data=df, kind='reg' ,line_kws={'color': 'red'}, height=8)
```

> Create a regression jointplot to visualize relationship between income and house price.

---

### 🗺 **8–9. Scatterplot with Spatial Reference**

```python
gdf = gpd.read_file('ca_state.zip')
gdf.to_crs(4326,inplace=True)
```

> Load California boundary shapefile and convert to standard CRS (EPSG:4326).

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

> Plot housing data as a georeferenced scatterplot over California map.

---

### 💰 **10. Categorize Income**

```python
q1 = df['median_income'].quantile(0.25)
q2 = df['median_income'].quantile(0.5)
q3 = df['median_income'].quantile(0.75)
at95 = df['median_income'].quantile(0.95)
```

> Compute income thresholds for categorical binning.

```python
df['income_cat'] = pd.Series(dtype='object')
```

> Create empty column for income category.

```python
# Assign income categories based on quartiles
df.loc[df['median_income']<q1,'income_cat'] = 'Low'
df.loc[(df['median_income']>=q1)&(df['median_income']<q2),'income_cat'] = 'Below average'
df.loc[(df['median_income']>=q2)&(df['median_income']<q3),'income_cat'] = 'Above average'
df.loc[(df['median_income']>=q3)&(df['median_income']<at95),'income_cat'] = 'High'
df.loc[df['median_income']>=at95,'income_cat'] = 'Very high'
```

> Label each row with appropriate income category.

---

### 📊 **11. Histogram of Income Category by Ocean Proximity**

```python
plt.figure(figsize=(12, 8))
sns.histplot(data=df.sort_values(by='median_income'),x='income_cat',hue='ocean_proximity',multiple='dodge')
plt.grid(True, alpha=0.6,axis='y')
```

> Visualize income category distributions, broken down by proximity to ocean.

---

### 📉 **12. Bar Plot: Income Category vs House Value**

```python
plt.figure(figsize=(12, 8))
plt.grid(True, alpha=0.6,axis='y')
sns.barplot(x='income_cat',y='median_house_value',data=df.sort_values(by='median_income'))
```

> Compare average house value across income categories.

---

### 🌊 **13. Bar Plot: Ocean Proximity vs House Value**

```python
plt.figure(figsize=(12, 8))
sns.barplot(x='ocean_proximity',y='median_house_value',data=df)
plt.grid(True, alpha=0.6,axis='y')
```

> Analyze how house value varies with proximity to ocean.

---

### 🔥 **14. Heatmap: Income Category vs Ocean Proximity**

```python
# Group and compute mean house values by category
heatmap_df = df[['ocean_proximity','income_cat','median_house_value']]
groups = heatmap_df.groupby(['ocean_proximity','income_cat'])
df_list = []
for i in groups.groups:
    df_list.append([i[0],i[1],groups.get_group(i)['median_house_value'].mean()])
heatmap_df = pd.DataFrame(df_list,columns=['ocean_proximity','income_cat','median_house_value'])
heatmap_df = heatmap_df.pivot('income_cat', 'ocean_proximity','median_house_value')
heatmap_df.drop(columns=['ISLAND'],inplace=True)
heatmap_df.sort_values(by=['<1H OCEAN','INLAND','NEAR BAY','NEAR OCEAN'],inplace=True)

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_df, annot=True, fmt=".0f", linewidths=0.5)
```

> Create a heatmap to show average house values based on both income and location.

---

### 🔁 **15. Reload Data and Fill Missing Values**

```python
df = pd.read_csv('housing.csv')
mean = df['total_bedrooms'].mean()
df['total_bedrooms'] = df['total_bedrooms'].fillna(mean)

median = df['total_rooms'].median()
df['total_rooms'] = df['total_rooms'].fillna(median)
```

> Reload dataset and fill missing values with mean/median.

> (KNN Imputer code is provided but commented out.)

---

### 📦 **16. Identify Outliers**

```python
num_df = df.select_dtypes(np.number)

# Generate boxplots to visually inspect outliers
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

> Plot boxplots for numeric columns to detect outliers.

---

### ❌ **17. Remove Outliers with IQR Rule**

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

df_droped_outlier = drop_outliers(df)
```

> Define and apply a function to remove outliers using the 1.5\*IQR rule.

```python
# Boxplots before and after removing outliers
nrows = len(num_df.columns)
fig, axes = plt.subplots(nrows, 2, figsize=(15, 30))
for i, column in enumerate(num_df.columns):
    sns.boxplot(x=df[column], ax=axes[i, 0]) 
    axes[i, 0].set_title(f'{column}',fontsize=14, fontweight='bold')
    sns.boxplot(x=df_droped_outlier[column], ax=axes[i, 1])  
    axes[i, 1].set_title(f'{column}',fontsize=14, fontweight='bold')    
fig.text(0.25, 0.95, 'Left: Original Data', ha='center', va='center', fontsize=16, fontweight='bold')
fig.text(0.75, 0.95, 'Right: Data with Outliers Removed', ha='center', va='center', fontsize=16, fontweight='bold')
plt.tight_layout()
fig.subplots_adjust(top=0.92, hspace=0.5)
```

> Compare before and after removing outliers using side-by-side boxplots.

---

### 🧠 **18–22. Machine Learning Preprocessing**

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
```

> Import preprocessing and transformation tools.

```python
X = df.drop(columns=['longitude','latitude','median_house_value']) 
y = df['median_house_value']
```

> Prepare features `X` and target `y` for prediction. Remove spatial columns.

```python
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
```

> Split data into training and testing sets.

```python
scaler = MinMaxScaler()
encoder = OneHotEncoder(handle_unknown='ignore')
```

> Initialize scaler and encoder for normalization and categorical encoding.

```python
num = X.select_dtypes(np.number)
cat = X.select_dtypes(exclude=[np.number])
```

> Separate numeric and categorical columns.

```python
processor = ColumnTransformer([
    ('scaler', scaler, num.columns),
    ('encoder', encoder, cat.columns)
])
```

> Build a combined transformer to apply scaler and encoder.

```python
processed_X_train = processor.fit_transform(X_train)
processed_X_test = processor.transform(X_test)
processed_y_train = scaler.fit_transform(np.array(y_train.to_numpy().reshape(-1,1)))
processed_y_test  = scaler.transform(y_test.to_numpy().reshape(-1,1))
```

> Apply transformations to both training and test sets.

---

Let me know if you want the markdown version with comments added directly inline.
