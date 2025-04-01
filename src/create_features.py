import pandas as pd
import numpy as np 
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile, chi2

col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'y']

# Load data
train_data = pd.read_csv('data/adult.data', names=col_names)
test_data = pd.read_csv('data/adult.test', names=col_names)

# Encode and impute values for target variable
train_y, test_y = train_data['y'], test_data['y']
train_y = train_y.map( {' >50K':1, ' >50K.':1, ' <=50K.':0, ' <=50K':0} )
test_y = test_y.map( {' >50K':1, ' >50K.':1, ' <=50K.':0, ' <=50K':0} )

train_y = train_y.values.reshape((-1,1))
test_y = test_y.values.reshape((-1,1))

impy = SimpleImputer(strategy="most_frequent")
impy.fit(train_y)
train_y = impy.transform(train_y)
test_y = impy.transform(test_y)

# Drop target variable 
train_data = train_data.drop(columns = ['y'])
test_data = test_data.drop(columns = ['y'])

# Create pipeline for imputing and scaling numeric variables
# one-hot encoding categorical variables, and select features based on chi-squared value
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("selector", SelectPercentile(chi2, percentile=50)),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, make_column_selector(dtype_include = ['int', 'float'])),
        ("cat", categorical_transformer, make_column_selector(dtype_exclude = ['int', 'float'])),
    ]
)

clf = Pipeline(
    steps=[("preprocessor", preprocessor)]
)

# Create new train and test data using the pipeline
clf.fit(train_data, train_y)
train_new = clf.transform(train_data)
test_new = clf.transform(test_data)

# Transform to dataframe and save as a csv
train_new = pd.DataFrame.sparse.from_spmatrix(train_new)
test_new = pd.DataFrame.sparse.from_spmatrix(test_new)
train_new['y'] = train_y
test_new['y'] = test_y

train_new.to_csv('data/processed_train_data.csv')
test_new.to_csv('data/processed_test_data.csv')

# Save pipeline
with open('data/pipeline.pkl','wb') as f:
    pickle.dump(clf,f)