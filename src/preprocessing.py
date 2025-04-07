import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import yaml

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params['preprocessing']

def load_wine_data(data_path):
    """
    Load the wine dataset from the specified path.
    
    Args:
        data_path (str): Path to the wine.data file
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector
    """
    # Define column names for the wine dataset
    column_names = ['class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash',
                   'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
                   'proanthocyanins', 'color_intensity', 'hue', 
                   'od280/od315_of_diluted_wines', 'proline']
    
    # Load the data
    df = pd.read_csv(data_path, header=None, names=column_names)
    
    # Separate features and target
    y = df['class']
    X = df.drop('class', axis=1)
    
    return X, y

def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split the data into training, validation, and test sets.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        test_size (float): Proportion of data to use for test set
        val_size (float): Proportion of remaining data to use for validation set
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: separate test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_data(X_train, X_val, X_test):
    """
    Scale the features using StandardScaler.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_val (pd.DataFrame): Validation features
        X_test (pd.DataFrame): Test features
        
    Returns:
        tuple: (X_train_scaled, X_val_scaled, X_test_scaled)
    """
    scaler = StandardScaler()
    
    # Fit scaler on training data and transform all sets
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    return X_train_scaled, X_val_scaled, X_test_scaled

def save_data(X_train_scaled, X_val_scaled, X_test_scaled, 
              y_train, y_val, y_test, save_dir='save_data'):
    """
    Save the processed datasets as parquet files.
    
    Args:
        X_train_scaled (pd.DataFrame): Scaled training features
        X_val_scaled (pd.DataFrame): Scaled validation features
        X_test_scaled (pd.DataFrame): Scaled test features
        y_train (pd.Series): Training target
        y_val (pd.Series): Validation target
        y_test (pd.Series): Test target
        save_dir (str): Directory to save the data
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save feature sets
    X_train_scaled.to_parquet(f'{save_dir}/x_train.parquet')
    X_val_scaled.to_parquet(f'{save_dir}/x_val.parquet')
    X_test_scaled.to_parquet(f'{save_dir}/x_test.parquet')
    
    # Save target sets
    pd.DataFrame(y_train, columns=['class']).to_parquet(f'{save_dir}/y_train.parquet')
    pd.DataFrame(y_val, columns=['class']).to_parquet(f'{save_dir}/y_val.parquet')
    pd.DataFrame(y_test, columns=['class']).to_parquet(f'{save_dir}/y_test.parquet')

def main():
    """Main function to run the preprocessing pipeline"""
    # Load parameters
    params = load_params()
    
    # Load data
    X, y = load_wine_data("data/wine/wine.data")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y,
        test_size=params['test_size'],
        val_size=params['val_size'],
        random_state=params['random_state']
    )
    
    # Scale data
    X_train_scaled, X_val_scaled, X_test_scaled = scale_data(
        X_train, X_val, X_test
    )
    
    # Save data
    save_data(
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test,
        save_dir=params['save_dir']
    )
    
    print("Data preprocessing completed successfully!")
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Validation set shape: {X_val_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")

if __name__ == "__main__":
    main()
