
import xgboost
import shap
import pandas as pd
import numpy as np
import os
from flask import render_template

def extract_shap_force_plots(bst: xgboost.Booster, X_test: pd.DataFrame, output_dir: str):
    """
    Calculates SHAP values for the given data and saves an interactive 
    SHAP force plot (HTML) for every row/customer in X_test.

    Args:
        bst (xgboost.Booster): The loaded XGBoost model booster object.
        X_test (pd.DataFrame): The DataFrame of features for prediction.
        output_dir (str): The directory path where the HTML files should be saved.
    
    Returns:
        list: A list of filenames (relative to output_dir) created.
    """
    print("Calculating SHAP values...")
    try:
        explainer = shap.TreeExplainer(bst)
        shap_values = explainer.shap_values(X_test)
        base_value = explainer.expected_value
        print("SHAP values calculated successfully.")
    except Exception as e:
        print(f"Error calculating SHAP values: {e}")
        return []

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving SHAP force plots to: {output_dir}")
    
    plot_filenames = []
    
    for i in range(X_test.shape[0]):
        plot_filename = f"force_plot_customer_{i}.html"
        full_plot_path = os.path.join(output_dir, plot_filename)
        
        try:
            # Generate the interactive force plot object
            # link='identity' is used to ensure the plot is correctly displayed in HTML
            html_plot = shap.force_plot(
                base_value, 
                shap_values[i], 
                X_test.iloc[i], 
                link='identity',
                show=False # Prevent immediate display
            )
            # shap_html = f"<head>{shap.getjs()}</head><body>{html_plot.html()}</body>"
            
            # Use shap.save_html to save the plot as a self-contained HTML file
            # shap.save_html(full_plot_path, shap_html)

            plot_filenames.append(plot_filename)
            # return render_template('shap_plots.html', shap_html=shap_html)
            shap.save_html(full_plot_path, html_plot)
            full_data=shap.force_plot(
                explainer.expected_value, shap_values[:1000, :], X_test.iloc[:1000, :]
            )
            shap.save_html(os.path.join(output_dir, "summary.html"), full_data)

        except Exception as e:
            print(f"Error saving plot for customer {i}: {e}")



if __name__ == '__main__':
    # --- Example Usage ---
    
    # NOTE: You must have a pre-trained model and the feature data file to run this.
    MODEL_PATH = './ml/xgb_churn_model.json'
    INPUT_DATA_PATH = 'data/x_processed.csv'
    SHAP_PLOTS_DIR = 'shap_plots/force_plots'

    # --- 1. Load Model ---
    try:
        bst = xgboost.Booster()
        bst.load_model(MODEL_PATH)
        expected_features = bst.feature_names
        print("XGBoost model loaded.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit()
        
    # --- 2. Load and Prepare Data (Simplified: Assumes x_processed.csv is the final feature set) ---
    try:
        X_test = pd.read_csv(INPUT_DATA_PATH)
        # Ensure column alignment with the model's expectations (essential step!)
        for col in expected_features:
            if col not in X_test.columns:
                X_test[col] = 0
        X_test = X_test[expected_features]
        print(f"Data loaded and prepared ({X_test.shape[0]} rows).")

    except Exception as e:
        print(f"Failed to load/prepare data: {e}")
        exit()

    # --- 3. Call the new function ---
    paths = extract_shap_force_plots(bst, X_test, SHAP_PLOTS_DIR)
    
    if paths:
        print("\nFirst 5 generated plot paths:")
        for path in paths[:5]:
            print(f"- {os.path.join(SHAP_PLOTS_DIR, path)}")
