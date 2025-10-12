import xgboost
import shap
import pandas as pd
import numpy as np
import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Tuple, Optional, List, Any

# --- CONFIGURATION ---
MODEL_PATH = './ml/xgb_churn_model.json'
INPUT_DATA_PATH = 'data/x_processed.csv'
# PREPARED_DATA_PATH is commented out as we are using in-memory dict for this test
GEMINI_MODEL_ID = "gemini-2.5-flash"
N_TOP_FEATURES = 5 # Number of features to send to the LLM

# --- 0. Setup and Utilities ---

def configure_gemini() -> Tuple[bool, Optional[genai.GenerativeModel]]:
    """Loads API key and configures the Gemini client."""
    try:
        dotenv_path = '.env'
        load_dotenv(dotenv_path)
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            # Only warn, as this configuration happens only if Task 2 is called
            print("Warning: GEMINI_API_KEY not found. On-demand strategy generation may fail.") 
            return False, None
        
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_ID)
        return True, gemini_model
    except Exception as e:
        print(f"Could not configure Gemini API: {e}")
        return False, None

def load_and_align_data(input_path: str, bst: xgboost.Booster) -> Optional[pd.DataFrame]:
    """
    Loads the already processed feature set (X_test) and performs structural 
    alignment (column order/presence) required by the XGBoost model. 
    No actual data transformation (scaling, encoding) is performed here.
    """
    try:
        X_test = pd.read_csv(input_path)
        print(f"Input feature data loaded from {input_path} ({X_test.shape[0]} rows).")
    except FileNotFoundError:
        print(f"Error: {input_path} not found.")
        return None

    # CRITICAL ALIGNMENT: Ensure feature set matches the model's expected features
    expected_features = bst.feature_names
    
    # Add any missing one-hot encoded columns if they weren't present in this batch
    for col in expected_features:
        if col not in X_test.columns:
            X_test[col] = 0
            
    # Re-order columns to match the model's expectation
    X_test = X_test[expected_features]
    return X_test

def calculate_all_shap_values(bst: xgboost.Booster, X_test: pd.DataFrame) -> np.ndarray:
    """Calculates and returns SHAP values for the entire test set."""
    print("Calculating SHAP values for all rows...")
    explainer = shap.TreeExplainer(bst)
    shap_values = explainer.shap_values(X_test)
    return shap_values

# --- TASK 1: PRE-COMPUTE AND STORE ALL ML DATA ---

def task1_prepare_and_store_all_ml_data() -> Optional[dict]:
    """
    Executes the offline, pre-calculation phase. Loads model, makes predictions,
    calculates SHAP values for ALL customers, and returns them for storage/use.
    """
    print("\n--- TASK 1: OFFLINE DATA PREPARATION (Pre-computing SHAP and Predictions) ---")
    
    # 1. Load Model
    try:
        bst = xgboost.Booster()
        bst.load_model(MODEL_PATH)
        print("XGBoost model loaded successfully.")
    except xgboost.core.XGBoostError as e:
        print(f"Error loading model: {e}")
        return None

    # 2. Load and Align Data
    # Renamed from preprocess to align
    X_test = load_and_align_data(INPUT_DATA_PATH, bst)
    if X_test is None:
        return None
    
    # 3. Make Predictions
    dtest = xgboost.DMatrix(X_test)
    predictions = bst.predict(dtest)

    # 4. Calculate SHAP Values
    shap_values = calculate_all_shap_values(bst, X_test)
    
    # 5. Package Data (simulating storage in a database/file system)
    prepared_data = {
        'X_test': X_test,
        'predictions': predictions,
        'shap_values': shap_values
    }

    print("SHAP values calculated and stored in 'prepared_data' dictionary.")
    return prepared_data

# --- TASK 2: ON-DEMAND GEMINI CALL ---

def task2_generate_strategy_on_demand(
    customer_id_or_index: int, 
    prepared_data: dict
) -> str:
    """
    Triggers a single, lightweight call to the Gemini API for a specific
    customer, using the pre-calculated SHAP values.
    """
    
    # 0. Setup and check API key
    use_gemini, gemini_model = configure_gemini()
    if not use_gemini or gemini_model is None:
        return "Gemini API not configured. Cannot generate AI strategy."

    # 1. Retrieve Data for the specific customer
    i = customer_id_or_index
    X_test = prepared_data['X_test']
    shap_values = prepared_data['shap_values']
    predictions = prepared_data['predictions']
    
    if i >= X_test.shape[0]:
        return f"Error: Customer index {i} out of bounds."
        
    print(f"\n--- TASK 2: ON-DEMAND GEMINI CALL for Customer Index {i} ---")

    # 2. Extract SHAP and Feature data for row i
    feature_names = X_test.columns
    shap_vals_row = shap_values[i]
    prediction = predictions[i]
    
    feature_contributions = sorted(
        zip(feature_names, shap_vals_row), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )
    top_features = feature_contributions[:N_TOP_FEATURES]
    
    # 3. Build Prompt Details
    prompt_details = []
    for feature, shap_val in top_features:
        customer_value = X_test.iloc[i][feature]
        
        # Formatting for clarity
        if feature.startswith('city_') or feature.startswith('marital_status_'):
            customer_value = "YES" if customer_value == 1 else "NO"
        
        direction = "INCREASES" if shap_val > 0 else "DECREASES"
        prompt_details.append(
            f"- Feature '{feature}': Customer's value is '{customer_value}'. This {direction} churn risk (SHAP: {shap_val:.4f})."
        )

    formatted_details = '\n'.join(prompt_details)
    
    # 4. Construct Prompt
    prompt = f"""
    Context: I am a retention specialist at a leading auto insurance company. A policyholder has been identified by our XGBoost model as having a high risk of not renewing their policy (lapsing).

    Policy Lapse Risk Score: {prediction:.4f}

    Top {N_TOP_FEATURES} Factors Influencing this Prediction (from SHAP analysis):
    {formatted_details}

    Task: Based *only* on the top {N_TOP_FEATURES} factors, provide **one** precise strategy to encourage this policyholder to renew. The response must be **less than 50 words** and follow this exact format:

    * Insight: [Explain the key insight]
    Action: [Propose a specific, concrete action]
    """
    
    # 5. Call Gemini
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error processing row {i} with Gemini: {e}")
        return f"ERROR: Gemini API call failed for customer {i}. Check logs."

# --- TASK 3: DRIVER CODE TO TEST ONCE ---

if __name__ == "__main__":
    
    # --- STEP 1: Execute Task 1 (Offline Batch Pre-computation) ---
    prepared_data = task1_prepare_and_store_all_ml_data()
    
    if prepared_data is None:
        print("FATAL: ML Data preparation failed. Exiting driver.")
    else:
        # --- STEP 2: Execute Task 2 (Simulated Frontend Clicks/API Calls) ---
        
        # Scenario A: User clicks "AI Insights" for the 1st customer (index 0)
        customer_index_a = 0
        strategy_a = task2_generate_strategy_on_demand(customer_index_a, prepared_data)
        
        print("\n=============================================")
        print(f"FINAL STRATEGY FOR CUSTOMER {customer_index_a} (Index 0):")
        print("=============================================")
        print(strategy_a)
        
        # Scenario B: User clicks "AI Insights" for the 5th customer (index 4)
        # We check if shape[0] (total rows) is greater than 4 to ensure index 4 (the 5th row, due to 0-indexing) exists.
        if prepared_data['X_test'].shape[0] > 4:
            customer_index_b = 85
            strategy_b = task2_generate_strategy_on_demand(customer_index_b, prepared_data)
            
            print("\n=============================================")
            print(f"FINAL STRATEGY FOR CUSTOMER {customer_index_b} (Index 4):")
            print("=============================================")
            print(strategy_b)
        else:
             print("\nSkipping second demonstration as data size is too small.")
