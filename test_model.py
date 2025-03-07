import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model

# Mock data for testing
@pytest.fixture
def mock_data():

    data = {
        'State': ['CA', 'TX', 'NV', 'TX', 'CA', 'NV'],
        'International plan': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
        'Voice mail plan': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes'],
        'Churn': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
        'Account length': [100, 200, 150, 180, 210, 160],
        'Area code': [408, 510, 510, 408, 408, 510],
        'Total day minutes': [100, 200, 150, 180, 210, 160],
        'Total day calls': [20, 30, 25, 22, 28, 30],
    }
    df = pd.DataFrame(data)
    df.to_csv('mock_train.csv', index=False)
    df.to_csv('mock_test.csv', index=False)
    return 'mock_train.csv', 'mock_test.csv'

# Test prepare_data function
def test_prepare_data(mock_data):
    train_path, test_path = mock_data
    X_train, X_test, y_train, y_test = prepare_data(train_path, test_path)
    
    # Check that data has been split and transformed correctly
    assert X_train.shape[0] == 4  # 80% of 6 data points
    assert X_test.shape[0] == 2  # 20% of 6 data points
    assert X_train.shape[1] > 0  # Check that features are present after transformation
    assert set(y_train).issubset({0, 1})  # Check that labels are encoded correctly

# Test train_model function
def test_train_model(mock_data):
    train_path, test_path = mock_data
    X_train, X_test, y_train, y_test = prepare_data(train_path, test_path)
    
    model = train_model(X_train, y_train)
    
    # Check that the model has been trained
    assert hasattr(model, 'predict')  # The model should have a 'predict' method after training

# Test evaluate_model function (integration test)
def test_evaluate_model(mock_data):
    train_path, test_path = mock_data
    X_train, X_test, y_train, y_test = prepare_data(train_path, test_path)
    
    model = train_model(X_train, y_train)
    
    # Evaluate the model and capture the printed output
    # You might want to mock the print statements or capture the output if needed.
    # For now, just check that it runs without errors
    evaluate_model(model, X_test, y_test)

# Test save_model and load_model functions
def test_save_load_model(mock_data):
    train_path, test_path = mock_data
    X_train, X_test, y_train, y_test = prepare_data(train_path, test_path)
    
    model = train_model(X_train, y_train)
    
    # Save the model
    save_model(model, 'test_model.pkl')
    
    # Load the model
    loaded_model = load_model('test_model.pkl')
    
    # Ensure that the loaded model is the same as the trained one
    assert hasattr(loaded_model, 'predict')  # The loaded model should have the 'predict' method


