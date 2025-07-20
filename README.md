# 🧠 Annotated Python Notebook

### 🔹 Step 1: Code Explanation
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
```

## 1. Load the housing.csv file using pandas.

### 🔹 Step 2: Code Explanation
```python
df = pd.read_csv('housing.csv')
```

## 2. Drop all rows with at least one missing value.

### 🔹 Step 3: Code Explanation
```python
df.dropna(inplace=True)
```

## 3. rooms_per_household

### 🔹 Step 4: Code Explanation
```python
df['rooms_per_household'] = df['total_rooms']/df['households']
```

## 4. rooms_per_householdpopulation_per_household

### 🔹 Step 5: Code Explanation
```python
df['population_per_household'] = df['total_rooms']/df['population']
```

## 5. Calculate the average house value for houses that are 50 years old or less. 

### 🔹 Step 6: Code Explanation
```python
housing_median_age_lessEqual50= df.loc[(df['housing_median_age']<=50),'median_house_value']
housing_median_age_lessEqual50.mean()
```

## 6. Calculate the correlation of “median_house_value”

### 🔹 Step 7: Code Explanation
```python
median_house_correlation = df.corr()[['median_house_value']]
```

### 🔹 Step 8: Code Explanation
```python
median_house_correlation
```

### 🔹 Step 9: Code Explanation
```python
median_house_correlation.sort_values(by='median_house_value',inplace=True)
```

### 🔹 Step 10: Code Explanation
```python
plt.figure(figsize=(4, 8))
sns.heatmap(median_house_correlation, annot=True, cmap='rocket')
plt.title('\'median_house_value\' Correlation Matrix Heatmap',pad=20,fontsize=15)
```

### Which factors seem to influence house prices?

> The greatest feature is influences **'median_house_value'** is **'median_income'** which is related to as posisitive correlation followed it the **'population_per_households'** and the **'rooms_per_households'** with fewer positive corrlation.

>The other features don't have corrleatiions with **'median_house_value'**.

>**'latitude'** and **'longtitude'** are spatial data.

>I don't see a negative correlation.

## 7. Create a Seaborn regression plot (jointplot) with income in the a-axis and house value on the y-axis

### 🔹 Step 11: Code Explanation
```python
plt.figure(figsize=(12, 8))
sns.jointplot(x='median_income', y='median_house_value', data=df, kind='reg' ,line_kws={'color': 'red'}, height=8)
```

## 8. Create the following scatterplot:
 - Longitude on x-axis
 - Latitude on y-axis
 - Size of datapoints is determined by population
 - Color of datapoints is determined by median_house_value

## 9. Use geo_pandas to draw the previous scatter plot on California map.

### 🔹 Step 12: Code Explanation
```python
gdf = gpd.read_file('ca_state.zip')
gdf.to_crs(4326,inplace=True)
```

### 🔹 Step 13: Code Explanation
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

## 10.Add the additional column “income_cat” with the following income categories:

### 🔹 Step 14: Code Explanation
```python
q1   = df['median_income'].quantile(0.25)
q2   = df['median_income'].quantile(0.5)
q3   = df['median_income'].quantile(0.75)
at95 = df['median_income'].quantile(0.95)
```

### 🔹 Step 15: Code Explanation
```python
df['income_cat'] = pd.Series(dtype='object')
```

### 🔹 Step 16: Code Explanation
```python
df.loc[df['median_income']<q1,'income_cat'] = 'Low'
df.loc[(df['median_income']>=q1)&(df['median_income']<q2),'income_cat'] = 'Below average'
df.loc[(df['median_income']>=q2)&(df['median_income']<q3),'income_cat'] = 'Above average'
df.loc[(df['median_income']>=q3)&(df['median_income']<at95),'income_cat'] = 'High'
df.loc[df['median_income']>=at95,'income_cat'] = 'Very high'
```

## 11. Create the following plot using seaborn:

### 🔹 Step 17: Code Explanation
```python
plt.figure(figsize=(12, 8))
sns.histplot(data=df.sort_values(by='median_income'),x='income_cat',hue='ocean_proximity',multiple='dodge')
plt.grid(True, alpha=0.6,axis='y')
```

## 12. Create the following plot using Seaborn

### 🔹 Step 18: Code Explanation
```python
plt.figure(figsize=(12, 8))
plt.grid(True, alpha=0.6,axis='y')
sns.barplot(x='income_cat',y='median_house_value',data=df.sort_values(by='median_income'))
```

## 13. Create the following plot using Seaborn

### 🔹 Step 19: Code Explanation
```python
plt.figure(figsize=(12, 8))
sns.barplot(x='ocean_proximity',y='median_house_value',data=df)
plt.grid(True, alpha=0.6,axis='y')
```

## 14. Create the following Seaborn Heatmap with mean house values for all combinations of income_cat and ocean_proximity.

### 🔹 Step 20: Code Explanation
```python
heatmap_df = df[['ocean_proximity','income_cat','median_house_value']]
```

### 🔹 Step 21: Code Explanation
```python
groups = heatmap_df.groupby(['ocean_proximity','income_cat'])
```

### 🔹 Step 22: Code Explanation
```python
df_list = []
for i in groups.groups:
    df_list.append([i[0],i[1],groups.get_group(i)['median_house_value'].mean()])
```

### 🔹 Step 23: Code Explanation
```python
heatmap_df = pd.DataFrame(df_list,columns=['ocean_proximity','income_cat','median_house_value'])
```

### 🔹 Step 24: Code Explanation
```python
heatmap_df = heatmap_df.pivot('income_cat', 'ocean_proximity','median_house_value')
```

### 🔹 Step 25: Code Explanation
```python
heatmap_df.drop(columns=['ISLAND'],inplace=True)
```

### 🔹 Step 26: Code Explanation
```python
heatmap_df.sort_values(by=['<1H OCEAN','INLAND','NEAR BAY','NEAR OCEAN'],inplace=True)
```

### 🔹 Step 27: Code Explanation
```python
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_df, annot=True, fmt=".0f", linewidths=0.5)
```

## 15.Reload the housing.csv file again, and handle missing data as the following:

### 🔹 Step 28: Code Explanation
```python
df = pd.read_csv('housing.csv')
```

- Missing values in the total_bedrooms column should be filled with the mean value.

### 🔹 Step 29: Code Explanation
```python
mean = df['total_bedrooms'].mean()
df['total_bedrooms'] = df['total_bedrooms'].fillna(mean)
```

### 🔹 Step 30: Code Explanation
```python
#from sklearn.impute import SimpleImputer 
#imputer = SimpleImputer(strategy='mean')
#df['total_bedrooms'] = imputer.fit_transform(df[['total_bedrooms']])
```

- Missing values in the total_rooms column should be filled with the median value.

### 🔹 Step 31: Code Explanation
```python
median = df['total_rooms'].median()
df['total_rooms'] = df['total_rooms'].fillna(median)
```

### 🔹 Step 32: Code Explanation
```python
#from sklearn.impute import SimpleImputer 
#imputer = SimpleImputer(strategy='median')
#df['total_rooms'] = imputer.fit_transform(df[['total_rooms']])
```

- Missing values in the total_rooms column should be filled with the value calculated by KNN algorithm, with k=2.

### 🔹 Step 33: Code Explanation
```python
#from sklearn.impute import KNNImputer
#knn_imputer = KNNImputer(n_neighbors=2)
#num_df = df.select_dtypes(np.number)
#df_imputed = pd.DataFrame(knn_imputer.fit_transform(num_df),columns=num_df.columns)
#df['total_rooms'] = df_imputed['total_rooms']
```

## 16. Identify columns that have outliers by using the boxplot of Seaborn or Matplotlib.

### 🔹 Step 34: Code Explanation
```python
num_df = df.select_dtypes(np.number)
```

### 🔹 Step 35: Code Explanation
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

## 17. Write python function to remove outliers from all columns by using 1.5*IQR rule.

### 🔹 Step 36: Code Explanation
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

### 🔹 Step 37: Code Explanation
```python
df_droped_outlier = drop_outliers(df)
```

### 🔹 Step 38: Code Explanation
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

## 18. We want to develop a model that should predict the median house value given the other attributes. Separate the median_house_value so that it can be used as the labels columns. Other columns will be used as features for machine learning.

### 🔹 Step 39: Code Explanation
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
```

### 🔹 Step 40: Code Explanation
```python
X = df.drop(columns=['longitude','latitude','median_house_value']) ##'latitude' and 'longtitude'** are spatial data.
y = df['median_house_value']
```

## 19. Split into train and test sets.

### 🔹 Step 41: Code Explanation
```python
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
```

## 20. Normalize numeric columns by using MinMaxScaler from sklearn.
## 21. Use OneHotEncoding to encode the ocean_proximity column. 
## 22. Use ColumnTransformer from sklearn to apply the previous twotransformation processes on the features.

### 🔹 Step 42: Code Explanation
```python
scaler = MinMaxScaler()
encoder = OneHotEncoder(handle_unknown='ignore')
```

### 🔹 Step 43: Code Explanation
```python
num = X.select_dtypes(np.number)
cat = X.select_dtypes(exclude=[np.number])
```

### 🔹 Step 44: Code Explanation
```python
processor = ColumnTransformer([
    ('scaler', scaler, num.columns),
    ('encoder', encoder, cat.columns)
])
```

### 🔹 Step 45: Code Explanation
```python
processed_X_train = processor.fit_transform(X_train)
processed_X_test = processor.transform(X_test)
```

### 🔹 Step 46: Code Explanation
```python
processed_y_train = scaler.fit_transform(np.array(y_train.to_numpy().reshape(-1,1)))
processed_y_test  = scaler.transform(y_test.to_numpy().reshape(-1,1))
```

## 23. Read the sarcasm.json file by using pandas

### 🔹 Step 47: Code Explanation
```python
df = pd.read_json('sarcasm.json')
```

## 24. Add the additional column “domain_name”. Extract the domain name from the article link.

### 🔹 Step 48: Code Explanation
```python
#def get_domin(string):
#    return string[:string.index('com')+3]
#df['domain_name'] = df['article_link'].apply(get_domin)
```

### 🔹 Step 49: Code Explanation
```python
from urllib.parse import urlparse
df['domain_name']= df['article_link'].apply(lambda x: urlparse(x).netloc)
```

## 25. Group headlines by the domain name. Plot bar plot showing the number of headlines per the domain name.

### 🔹 Step 50: Code Explanation
```python
domain_counts = df.groupby('domain_name').size().reset_index(name='headline_count')
domain_counts.sort_values('headline_count',inplace=True,ascending=False)
plt.figure(figsize=(16, 10))
ax = sns.barplot(y='domain_name', x='headline_count', data=domain_counts,color='tab:blue')
plt.title("Number of Headlines per Domain", fontsize=16, pad=15)
plt.xlabel("Number of Headlines",fontsize=16) 
plt.ylabel("Domain Name",fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
for p in ax.patches:
    width = p.get_width()
    ax.text(width + 0.2, p.get_y() + p.get_height() / 2, f'{int(width)}', ha='left', va='center', fontsize=15)
```

## 26. Apply the following preprocessing steps on the headlines in sequence:

### 🔹 Step 51: Code Explanation
```python
import nltk
from nltk.tokenize import word_tokenize,wordpunct_tokenize
```

### 🔹 Step 52: Code Explanation
```python
txt = df['headline'].tolist()
```

### 🔹 Step 53: Code Explanation
```python
nltk.download('punkt')
```

### 🔹 Step 54: Code Explanation
```python
nltk.download('stopwords')
```

### 🔹 Step 55: Code Explanation
```python
import string
punctuations = string.punctuation+'0123456789'
```

### 🔹 Step 56: Code Explanation
```python
stopwords = nltk.corpus.stopwords.words('english')
```

### 🔹 Step 57: Code Explanation
```python
tokenized = [wordpunct_tokenize(d.lower()) for d in txt]
```

### 🔹 Step 58: Code Explanation
```python
cleaned = []
for i in tokenized:
    cleaned.append([s for s in [t for t in i if t not in stopwords] if s not in punctuations])
```

### 🔹 Step 59: Code Explanation
```python
from nltk.stem.snowball import SnowballStemmer
```

### 🔹 Step 60: Code Explanation
```python
stemmer = SnowballStemmer(language='english')
```

### 🔹 Step 61: Code Explanation
```python
steamed = []
for i in cleaned:
    steamed.append(' '.join([stemmer.stem(t) for t in i]))
```

### 🔹 Step 62: Code Explanation
```python
from sklearn.feature_extraction.text import TfidfVectorizer
```

### 🔹 Step 63: Code Explanation
```python
vectorizer = TfidfVectorizer(max_features=2000)
```

### 🔹 Step 64: Code Explanation
```python
matrix = vectorizer.fit_transform(steamed)
```

### 🔹 Step 65: Code Explanation
```python
matrix_array = matrix.toarray()
matrix_array
```

### 🔹 Step 66: Code Explanation
```python
words = vectorizer.get_feature_names()
for word,weight in zip(words, matrix_array.T):
    print(f"Word: \'{word}\' | weight: {weight}")
```

### 🔹 Step 67: Code Explanation
```python
##Yasser I Barhoom (Geomatics Eng.)
```

### 🔹 Step 68: Code Explanation
```python

```

