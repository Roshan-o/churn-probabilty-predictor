import xgboost
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import google.generativeai as genai
from dotenv import load_dotenv

# --- 0. Load API Key and Configure Gemini ---
try:
    # Path to your .env file
    dotenv_path = '/home/mohith/Documents/mega2/Archive/.env'
    load_dotenv(dotenv_path)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not found in .env file or is empty.")
        use_gemini = False
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        print("Gemini API configured successfully.")
        use_gemini = True
except Exception as e:
    print(f"Could not configure Gemini API: {e}")
    use_gemini = False


# --- 1. Load your pre-trained XGBoost model ---
model_path = 'xgb_churn_model.json' 
try:
    bst = xgboost.Booster()
    bst.load_model(model_path)
    print("XGBoost model loaded successfully.")
except xgboost.core.XGBoostError as e:
    print(f"Error loading model: {e}")
    exit()

# --- 2. Prepare your data for prediction ---
try:
    test_df_raw = pd.read_csv('/home/mohith/Documents/mega2/Archive/data/x_test.csv')
    print("x_test.csv loaded successfully.")
except FileNotFoundError:
    print("Error: /home/mohith/Documents/mega2/Archive/data/x_test.csv not found.")
    exit()

# Preprocess the data
test_df_processed = pd.get_dummies(test_df_raw, columns=['city', 'marital_status'], dummy_na=False)

def parse_home_value(value):
    if isinstance(value, str):
        return int(value.split(' ')[0].replace(',', ''))
    return np.nan

test_df_processed['home_market_value'] = test_df_raw['home_market_value'].apply(parse_home_value)
test_df_processed['home_market_value'].fillna(test_df_processed['home_market_value'].median(), inplace=True)

expected_features = bst.feature_names
for col in expected_features:
    if col not in test_df_processed.columns:
        test_df_processed[col] = 0
X_test = test_df_processed[expected_features]

print(f"\nMaking predictions on data with {X_test.shape[0]} rows.")

# --- 3. Make predictions ---
dtest = xgboost.DMatrix(X_test)
predictions = bst.predict(dtest)
print("Predictions generated.")

# --- 4. Use SHAP to explain the model's predictions ---
print("\nCalculating SHAP values...")
explainer = shap.TreeExplainer(bst)
shap_values = explainer.shap_values(X_test)
print("SHAP values calculated.")

# --- 5. Visualize the explanations ---
print("\nGenerating SHAP summary plot...")
plt.figure(figsize=(10, 20))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("SHAP Feature Importance Summary")
plt.tight_layout()
plt.show()

print("\nGenerating SHAP waterfall plot for the first prediction...")
shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                     base_values=explainer.expected_value, 
                                     data=X_test.iloc[0],
                                     feature_names=X_test.columns),
                    show=False)
plt.tight_layout()
plt.show()

print("\nGenerating SHAP force plot for the first prediction...")
# The force plot provides another view of the feature contributions for a single prediction.
# We use matplotlib=True to render it correctly in a script environment.
shap.force_plot(explainer.expected_value, 
                shap_values[0,:], 
                X_test.iloc[0,:], 
                matplotlib=True, 
                show=False)
plt.title("SHAP Force Plot for First Prediction")
plt.tight_layout()
plt.show()


# --- 6. Get AI-powered recommendations for the first customer ---
if use_gemini:
    print("\n--- Generating Churn Prevention Recommendations with Gemini ---")
    
    # Get top 5 feature contributions
    feature_names = X_test.columns
    shap_vals_first = shap_values[0]
    feature_contributions = sorted(zip(feature_names, shap_vals_first), key=lambda x: abs(x[1]), reverse=True)
    top_5_features = feature_contributions[:5]
    
    # Build the prompt for the Gemini API
    prompt_details = []
    for feature, shap_val in top_5_features:
        customer_value = X_test.iloc[0][feature]
        direction = "increases" if shap_val > 0 else "decreases"
        prompt_details.append(f"- Feature '{feature}': Customer's value is '{customer_value}'. This {direction} churn risk (SHAP value: {shap_val:.4f}).")

    # Join the details into a single string first to avoid the f-string backslash error
    formatted_details = '\n'.join(prompt_details)

    prompt = f"""
    Context: I am a retention specialist at a leading auto insurance company like Chubb. A policyholder has been identified by our XGBoost model as having a high risk of not renewing their policy (lapsing).

    Policy Lapse Risk Score: {predictions[0]:.4f} (A higher score means a higher likelihood of non-renewal).

    Top 5 Factors Influencing this Prediction (from SHAP analysis):
    {formatted_details}

    Task: Based *only* on these top 5 factors, provide a bulleted list of 3 distinct strategies to encourage this policyholder to renew. For each strategy, provide one 'Description' explaining the insight from the data, and one 'Suggestion' for a concrete action to take.

    Format each strategy exactly as follows:
    * Strategy [Number]:
      Description: [Explain the insight based on the feature and its value]
      Suggestion: [Propose a specific action to take]
    """

    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        response = model.generate_content(prompt)
        print("\n--- Gemini Recommendations ---")
        print(response.text)
        print("--------------------------\n")
    except Exception as e:
        print(f"\nCould not get recommendations from Gemini: {e}")
else:
    print("\nSkipping Gemini recommendations because API is not configured.")

print("Script finished.")