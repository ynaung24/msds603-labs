import pandas as pd
import numpy as np 
import pickle
import yaml
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile, chi2

def load_data(train_name, test_name, col_names):
    # Load data
    train_data = pd.read_csv(train_name, names=col_names)
    test_data = pd.read_csv(test_name, names=col_names)
    return train_data, test_data

def process_data(train_data, test_data, chi2percentile):
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
            ("selector", SelectPercentile(chi2, percentile=chi2percentile)),
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
    return train_new, test_new, clf

def save_data(train_new, test_new, train_name, test_name, clf, clf_name):
    train_new.to_csv(train_name)
    test_new.to_csv(test_name)
    
    # Save pipeline
    with open(clf_name,'wb') as f:
        pickle.dump(clf,f)

if __name__=="__main__":
    
    params = yaml.safe_load(open("params.yaml"))["features"]
    train_name = params["train_path"]
    test_name = params["test_path"]
    chi2percentile = params["chi2percentile"]
    col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'y']

    train_data, test_data = load_data(train_name, test_name, col_names)
    train_new, test_new, clf = process_data(train_data, test_data, chi2percentile)
    save_data(train_new, test_new, 'data/processed_train_data.csv', 'data/processed_test_data.csv', clf, 'data/pipeline.pkl')