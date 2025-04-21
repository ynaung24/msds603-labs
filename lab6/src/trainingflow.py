from metaflow import FlowSpec, step, Parameter
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import sys

# Add the parent directory to the path to import preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import load_wine_data, split_data, scale_data, save_data

class WineClassifierFlow(FlowSpec):
    """
    A Metaflow flow for training wine classifier models.
    
    This flow:
    1. Ingests raw data
    2. Performs feature transformations
    3. Trains multiple models (Decision Tree, Random Forest, Logistic Regression)
    4. Evaluates and selects the best model
    5. Registers models using MLFlow
    """
    
    # Parameters for the flow
    test_size = Parameter('test_size', default=0.2, type=float)
    val_size = Parameter('val_size', default=0.2, type=float)
    random_state = Parameter('random_state', default=42, type=int)
    data_path = Parameter('data_path', default='../data/wine/wine.data', type=str)
    save_dir = Parameter('save_dir', default='../data/wine', type=str)
    
    @step
    def start(self):
        """
        Start step: Load and preprocess the data
        """
        print("Starting the Wine Classifier Flow")
        
        # Load the data
        self.X, self.y = load_wine_data(self.data_path)
        print(f"Data loaded with shape: X={self.X.shape}, y={self.y.shape}")
        
        # Split the data
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = split_data(
            self.X, self.y, 
            test_size=self.test_size, 
            val_size=self.val_size, 
            random_state=self.random_state
        )
        print(f"Data split: train={self.X_train.shape}, val={self.X_val.shape}, test={self.X_test.shape}")
        
        # Move to the next step
        self.next(self.transform_features)
    
    @step
    def transform_features(self):
        """
        Transform features step: Scale the features
        """
        print("Transforming features")
        
        # Scale the features
        self.X_train_scaled, self.X_val_scaled, self.X_test_scaled = scale_data(
            self.X_train, self.X_val, self.X_test
        )
        
        # Save the processed data
        save_data(
            self.X_train_scaled, self.X_val_scaled, self.X_test_scaled,
            self.y_train, self.y_val, self.y_test,
            save_dir=self.save_dir
        )
        
        print("Features scaled and data saved successfully")
        
        # Move to the next step
        self.next(self.train_decision_tree, self.train_random_forest, self.train_logistic_regression)
    
    @step
    def train_decision_tree(self):
        """
        Train Decision Tree model step
        """
        print("Training Decision Tree model")
        
        param_grid = {
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Create and train model
        model = DecisionTreeClassifier(random_state=self.random_state)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.model_name = 'decision_tree_wine'
        
        # Move to the next step
        self.next(self.join_models)
    
    @step
    def train_random_forest(self):
        """
        Train Random Forest model step
        """
        print("Training Random Forest model")
        
        # Define parameter grid for Random Forest
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Create and train model
        model = RandomForestClassifier(random_state=self.random_state)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.model_name = 'random_forest_wine'
        
        # Move to the next step
        self.next(self.join_models)
    
    @step
    def train_logistic_regression(self):
        """
        Train Logistic Regression model step
        """
        print("Training Logistic Regression model")
        
        # Define parameter grid for Logistic Regression
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'max_iter': [1000],
            'solver': ['lbfgs', 'liblinear']
        }
        
        # Create and train model
        model = LogisticRegression(random_state=self.random_state)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.model_name = 'logistic_regression_wine'
        
        # Move to the next step
        self.next(self.join_models)
    
    @step
    def join_models(self, inputs):
        """
        Join models step: Evaluate and register all models
        """
        print("Evaluating and registering models")
        
        # Get the scaled training data from the transform_features step
        transform_step = inputs[0]
        X_train_scaled = transform_step.X_train_scaled
        y_train = transform_step.y_train
        
        # Set up MLFlow with the running server
        mlflow.set_tracking_uri("http://localhost:5001")
        mlflow.set_experiment("wine-classifier")
        
        # Evaluate and register each model
        for inp in inputs:
            # Calculate metrics on training data
            y_pred = inp.model.predict(X_train_scaled)
            metrics = {
                'train_accuracy': accuracy_score(y_train, y_pred),
                'train_precision': precision_score(y_train, y_pred, average='weighted'),
                'train_recall': recall_score(y_train, y_pred, average='weighted'),
                'train_f1': f1_score(y_train, y_pred, average='weighted')
            }
            
            # Log model with MLFlow
            with mlflow.start_run(run_name=f"{inp.model_name}_run") as run:
                # Log parameters
                mlflow.log_params(inp.best_params)
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Log the model
                mlflow.sklearn.log_model(
                    sk_model=inp.model,
                    artifact_path="model",
                    registered_model_name=inp.model_name
                )
                
                print(f"Registered {inp.model_name} with training metrics: {metrics}")
                print(f"MLflow run ID: {run.info.run_id}")
        
        # Move to the next step
        self.next(self.end)
    
    @step
    def end(self):
        """
        End step: Finalize the flow
        """
        print("Wine Classifier Flow completed successfully")

if __name__ == '__main__':
    WineClassifierFlow()
