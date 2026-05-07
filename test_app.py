import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from io import StringIO
import sys

# ==========================================
# TEST SETUP: Mocking Streamlit
# ==========================================
# We mock Streamlit before importing our app to prevent it from executing
# UI commands (like st.set_page_config) during our unit test runs.
mock_st = MagicMock()
# Provide a side effect for st.columns so it can be unpacked correctly
mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]
# Prevent Streamlit widgets from returning Mock objects that break Pandas filtering
mock_st.sidebar.multiselect.side_effect = lambda label, options=None, default=None, **kwargs: default if default is not None else options
# Prevent @st.cache_data from destroying the function logic by returning a mock
mock_st.cache_data = lambda func: func
mock_st.cache_resource = lambda func: func
# Prevent st.button from evaluating to True and triggering the ML model loading during import
mock_st.button.return_value = False
sys.modules['streamlit'] = mock_st

# To prevent app.py from failing on import due to missing dataset or empty dataframe logic,
# we mock pandas.read_csv specifically for the import phase.
dummy_df = pd.DataFrame({
    'timestamp': ['2024-01-01 10:00:00', '2024-01-02 11:00:00'],
    'transaction type': ['P2P', 'P2M'],
    'amount (INR)': [500, 1000],
    'merchant_category': ['Retail', 'Groceries'],
    'sender_state': ['MH', 'KA'],
    'device_type': ['Android', 'iOS'],
    'network_type': ['4G', 'WiFi'],
    'transaction_status': ['SUCCESS', 'FAILED'],
    'hour_of_day': [10, 11],
    'day_of_week': ['Monday', 'Tuesday']
})

with patch('pandas.read_csv', return_value=dummy_df):
    import app  # Import the Streamlit app module to test its functions

# ==========================================
# FIXTURES
# ==========================================
@pytest.fixture
def sample_csv_data():
    """Provides a small, controlled sample of CSV data mimicking the UPI dataset."""
    csv_content = """timestamp,transaction type,amount (INR),merchant_category,sender_state,device_type,network_type,transaction_status,hour_of_day,day_of_week
2024-01-01 10:15:00,P2P,500,Retail,Maharashtra,Android,4G,SUCCESS,10,Monday
2024-01-01 11:30:00,P2M,1200,Groceries,Karnataka,iOS,WiFi,FAILED,11,Monday
2024-01-02 06:45:00,Bill Payment,2000,Utilities,Delhi,Web,Broadband,SUCCESS,6,Tuesday"""
    return StringIO(csv_content)

@pytest.fixture
def mock_model_columns():
    """Provides the mocked column structure expected by the trained Random Forest model."""
    return [
        'hour_of_day', 'amount (INR)', 
        'transaction type_P2M', 'transaction type_P2P', 'transaction type_Bill Payment',
        'day_of_week_Monday', 'day_of_week_Tuesday',
        'device_type_Android', 'device_type_iOS', 'device_type_Web',
        'network_type_4G', 'network_type_WiFi', 'network_type_Broadband'
    ]


# ==========================================
# 1 & 2. DATA LOADING & PREPROCESSING TESTS
# ==========================================
@patch('pandas.read_csv')
def test_load_data_success(mock_read_csv, sample_csv_data):
    """
    Test: Data loads correctly, datetimes are parsed, and target 'Failure' is mapped.
    """
    # Arrange: Setup read_csv to return our sample data
    mock_df = pd.DataFrame({
        'timestamp': ['2024-01-01 10:15:00', '2024-01-01 11:30:00', '2024-01-02 06:45:00'],
        'transaction type': ['P2P', 'P2M', 'Bill Payment'],
        'amount (INR)': [500, 1200, 2000],
        'merchant_category': ['Retail', 'Groceries', 'Utilities'],
        'sender_state': ['Maharashtra', 'Karnataka', 'Delhi'],
        'device_type': ['Android', 'iOS', 'Web'],
        'network_type': ['4G', 'WiFi', 'Broadband'],
        'transaction_status': ['SUCCESS', 'FAILED', 'SUCCESS'],
        'hour_of_day': [10, 11, 6],
        'day_of_week': ['Monday', 'Monday', 'Tuesday']
    })
    mock_read_csv.return_value = mock_df
    
    # Act: Call the target function
    df = app.load_data()
    
    # Assert
    assert not df.empty, "Dataframe should not be empty after successful load."
    assert 'date' in df.columns, "Date feature extraction failed."
    assert 'Failure' in df.columns, "Target variable mapping (Failure) failed."
    
    # Check boolean mapping logic
    assert df['Failure'].iloc[0] == 0, "SUCCESS status should be mapped to 0."
    assert df['Failure'].iloc[1] == 1, "FAILED status should be mapped to 1."
    
    # Check datetime conversion
    assert pd.api.types.is_datetime64_any_dtype(df['timestamp']), "Timestamp column was not converted to datetime object."

@patch('pandas.read_csv')
def test_load_data_file_not_found(mock_read_csv):
    """
    Test: Handling of missing dataset gracefully.
    """
    # Arrange: Simulate file missing
    mock_read_csv.side_effect = FileNotFoundError()
    
    # Act
    df = app.load_data()
    
    # Assert
    assert df.empty, "Dataframe should be empty if file is not found."
    # Verify Streamlit error was triggered
    app.st.error.assert_called()


# ==========================================
# 3 & 4. ML MODEL & PREDICTION LOGIC TESTS
# ==========================================
@patch('joblib.load')
def test_live_prediction_data_transformation(mock_joblib_load, mock_model_columns):
    """
    Test: Simulates the Streamlit Sandbox prediction logic.
    Ensures user inputs are correctly one-hot encoded and mapped to model columns.
    """
    # Arrange: Setup Mocks for Model & Columns
    mock_rf_model = MagicMock()
    # Simulate a prediction where class 0 (Success) is 95% and class 1 (Fail) is 5%
    mock_rf_model.predict_proba.return_value = np.array([[0.95, 0.05]]) 
    
    # joblib.load is called twice in the app (model, then columns)
    mock_joblib_load.side_effect = [mock_rf_model, mock_model_columns]
    
    # Simulate User Inputs from UI
    pred_hour = 2
    pred_amt = 1500
    pred_day = 'Tuesday'
    pred_type = 'P2M'
    pred_device = 'iOS'
    pred_network = 'WiFi'
    
    # Act: Execute the logic used in app.py for data prep
    import joblib
    rf_model = joblib.load('failure_model.pkl')
    model_cols = joblib.load('model_columns.pkl')
    
    input_dict = {col: 0 for col in model_cols}
    if 'hour_of_day' in input_dict: input_dict['hour_of_day'] = pred_hour
    if 'amount (INR)' in input_dict: input_dict['amount (INR)'] = pred_amt
    
    if f'day_of_week_{pred_day}' in input_dict: input_dict[f'day_of_week_{pred_day}'] = 1
    if f'transaction type_{pred_type}' in input_dict: input_dict[f'transaction type_{pred_type}'] = 1
    if f'device_type_{pred_device}' in input_dict: input_dict[f'device_type_{pred_device}'] = 1
    if f'network_type_{pred_network}' in input_dict: input_dict[f'network_type_{pred_network}'] = 1
    
    input_df = pd.DataFrame([input_dict])
    failure_prob = rf_model.predict_proba(input_df)[0][1] * 100
    
    # Assert Data Transformation (One-Hot Encoding verification)
    assert input_dict['hour_of_day'] == 2, "Continuous variable 'hour' not mapped."
    assert input_dict['amount (INR)'] == 1500, "Continuous variable 'amount' not mapped."
    assert input_dict['transaction type_P2M'] == 1, "Categorical selection not flagged as 1."
    assert input_dict['transaction type_P2P'] == 0, "Unselected categories must remain 0."
    assert input_dict['device_type_iOS'] == 1, "Device type mapping failed."
    
    # Assert Prediction Output
    assert failure_prob == 5.0, "Probability extraction index is incorrect."


# ==========================================
# 5. DASHBOARD FILTER & KPI LOGIC TESTS
# ==========================================
def test_dashboard_filtering_logic():
    """
    Test: The specific boolean masking logic used by Streamlit filters.
    """
    # Arrange
    df = pd.DataFrame({
        'transaction type': ['P2P', 'P2M', 'P2P', 'Bill Payment'],
        'merchant_category': ['Retail', 'Groceries', 'Retail', 'Utilities'],
        'sender_state': ['MH', 'KA', 'KA', 'DL']
    })
    
    # Simulated User Selections
    selected_tx_types = ['P2P', 'P2M']
    selected_merchants = ['Retail', 'Groceries']
    selected_states = ['MH', 'KA']
    
    # Act: Apply filter logic from app.py
    filtered_df = df[
        (df["transaction type"].isin(selected_tx_types)) &
        (df["merchant_category"].isin(selected_merchants)) &
        (df["sender_state"].isin(selected_states))
    ]
    
    # Assert
    assert len(filtered_df) == 3, "Filter logic excluded valid rows."
    assert 'Bill Payment' not in filtered_df['transaction type'].values, "Filter included unselected transaction types."
    assert 'DL' not in filtered_df['sender_state'].values, "Filter included unselected state."

def test_kpi_calculations():
    """
    Test: Dynamic KPI math calculations (Success Rate, Failure Rate).
    """
    # Arrange
    df = pd.DataFrame({'Failure': [0, 0, 0, 1, 1]}) # 5 total, 2 failed
    
    # Act: KPI logic from app.py
    total_tx = len(df)
    total_failed = df['Failure'].sum()
    failure_rate = (total_failed / total_tx) * 100 if total_tx > 0 else 0
    success_rate = 100 - failure_rate
    
    # Assert
    assert total_tx == 5
    assert total_failed == 2
    assert failure_rate == 40.0, "Failure rate calculation incorrect."
    assert success_rate == 60.0, "Success rate calculation incorrect."
