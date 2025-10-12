"""
SHAP Full Dataset Analysis for XGBoost Customer Churn Model

This script generates SHAP force plots for the entire dataset using your trained XGBoost model.
It creates individual HTML files for each customer and a summary visualization.
"""

import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import joblib
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_xgboost_model_and_data():
    """
    Load the trained XGBoost model and processed data
    """
    try:
        # Load the XGBoost model
        model = xgb.XGBClassifier()
        model.load_model("xgb_churn_model.json")
        print("✅ XGBoost model loaded successfully")
        
        # Load the scaler
        scaler = joblib.load("scaler.pkl")
        print("✅ Scaler loaded successfully")
        
        # Load the processed data
        X_data = pd.read_csv("data/x_processed.csv")
        print(f"✅ Processed data loaded: {X_data.shape}")
        
        return model, scaler, X_data
        
    except Exception as e:
        print(f"❌ Error loading model or data: {e}")
        return None, None, None

def generate_full_dataset_shap(model, X_data, max_customers=None, output_dir='force_plots'):
    """
    Generate SHAP force plots for the full dataset
    
    Args:
        model: Trained XGBoost model
        X_data: Processed feature data
        max_customers: Limit number of customers (None for all)
        output_dir: Directory to save HTML files
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Limit dataset if specified
    if max_customers:
        X_data_sample = X_data.head(max_customers)
        print(f"🎯 Generating SHAP for {max_customers} customers")
    else:
        X_data_sample = X_data
        print(f"🎯 Generating SHAP for all {len(X_data)} customers")
    
    try:
        # Create SHAP explainer
        print("🔄 Creating SHAP explainer...")
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values for the dataset
        print("🔄 Calculating SHAP values...")
        shap_values = explainer.shap_values(X_data_sample)
        
        # Get expected value
        expected_value = explainer.expected_value
        
        print(f"✅ SHAP values calculated. Shape: {shap_values.shape}")
        
        # Generate individual force plots
        print("🔄 Generating individual force plots...")
        
        successful_plots = 0
        failed_plots = 0
        
        for i in range(len(X_data_sample)):
            try:
                # Create force plot for this customer
                force_plot = shap.force_plot(
                    expected_value,
                    shap_values[i],
                    X_data_sample.iloc[i],
                    matplotlib=False,
                    show=False
                )
                
                # Get HTML content
                html_content = force_plot._repr_html_()
                
                # Enhanced HTML with proper styling and SHAP JS library
                enhanced_html = create_enhanced_shap_html(
                    customer_id=i,
                    shap_html=html_content,
                    customer_data=X_data_sample.iloc[i],
                    prediction=model.predict_proba(X_data_sample.iloc[i:i+1])[0][1],
                    expected_value=expected_value
                )
                
                # Save the HTML file
                output_file = Path(output_dir) / f'force_plot_customer_{i}.html'
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(enhanced_html)
                
                successful_plots += 1
                
                if i % 10 == 0:  # Progress update every 10 customers
                    print(f"   📊 Generated {i+1}/{len(X_data_sample)} plots...")
                    
            except Exception as e:
                print(f"   ⚠️ Failed to generate plot for customer {i}: {e}")
                failed_plots += 1
                continue
        
        print(f"\n✅ SHAP force plots generation complete!")
        print(f"   📊 Successful plots: {successful_plots}")
        print(f"   ❌ Failed plots: {failed_plots}")
        print(f"   📁 Output directory: {output_dir}")
        
        # Generate summary statistics
        generate_shap_summary(shap_values, X_data_sample, expected_value, output_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating SHAP plots: {e}")
        return False

def create_enhanced_shap_html(customer_id, shap_html, customer_data, prediction, expected_value):
    """
    Create enhanced HTML with proper styling and information
    """
    
    # Get top features (by absolute SHAP value)
    feature_info = []
    for feature, value in customer_data.items():
        feature_info.append(f"<li><strong>{feature}:</strong> {value:.3f if isinstance(value, (int, float)) else value}</li>")
    
    feature_list = "\\n".join(feature_info[:10])  # Show top 10 features
    
    enhanced_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>SHAP Analysis - Customer {customer_id}</title>
    <script src="https://cdn.jsdelivr.net/gh/slundberg/shap@master/js/dist/bundle.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 600;
        }}
        
        .header h2 {{
            margin: 10px 0 0 0;
            font-size: 1.5em;
            opacity: 0.9;
        }}
        
        .navigation {{
            margin-bottom: 20px;
        }}
        
        .btn {{
            display: inline-block;
            padding: 12px 24px;
            background: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 25px;
            font-weight: 600;
            transition: all 0.3s ease;
            margin-right: 10px;
        }}
        
        .btn:hover {{
            background: #2980b9;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            color: white;
            text-decoration: none;
        }}
        
        .prediction-card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            border-left: 5px solid #e74c3c;
        }}
        
        .prediction-score {{
            font-size: 2.5em;
            font-weight: bold;
            color: #e74c3c;
            text-align: center;
            margin-bottom: 10px;
        }}
        
        .shap-container {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            overflow-x: auto;
        }}
        
        .info-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }}
        
        .info-card {{
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }}
        
        .info-card h3 {{
            color: #2c3e50;
            margin-top: 0;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        
        .feature-list {{
            list-style: none;
            padding: 0;
        }}
        
        .feature-list li {{
            padding: 5px 0;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        .explanation {{
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
        }}
        
        .explanation h3 {{
            margin-top: 0;
        }}
        
        .explanation ul {{
            margin-bottom: 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 SHAP Force Plot Analysis</h1>
            <h2>Customer {customer_id}</h2>
        </div>
        
        <div class="navigation">
            <a href="/shap" class="btn">← Back to SHAP Home</a>
            <a href="/predictions" class="btn">📊 Predictions Dashboard</a>
            <a href="/customer-analysis" class="btn">👥 Customer Analysis</a>
        </div>
        
        <div class="prediction-card">
            <div class="prediction-score">{prediction:.1%}</div>
            <p style="text-align: center; margin: 0; font-size: 1.2em;">
                <strong>Churn Probability</strong><br>
                <small>Expected Value: {expected_value:.3f}</small>
            </p>
        </div>
        
        <div class="explanation">
            <h3>🔍 Understanding This SHAP Analysis</h3>
            <ul>
                <li><strong>Red features:</strong> Push the prediction towards higher churn probability</li>
                <li><strong>Blue features:</strong> Push the prediction towards lower churn probability</li>
                <li><strong>Bar width:</strong> Represents the magnitude of each feature's impact</li>
                <li><strong>Expected value:</strong> The average prediction across all customers</li>
            </ul>
        </div>
        
        <div class="shap-container">
            <h3 style="margin-top: 0; color: #2c3e50;">📈 Interactive SHAP Force Plot</h3>
            {shap_html}
        </div>
        
        <div class="info-cards">
            <div class="info-card">
                <h3>📋 Customer Features (Top 10)</h3>
                <ul class="feature-list">
                    {feature_list}
                </ul>
            </div>
            
            <div class="info-card">
                <h3>💡 Actionable Insights</h3>
                <ul>
                    <li>Focus retention efforts on red (risk-increasing) features</li>
                    <li>Leverage blue (protective) features in your strategy</li>
                    <li>Compare with other customers to identify patterns</li>
                    <li>Use these insights for personalized interventions</li>
                </ul>
            </div>
        </div>
    </div>
</body>
</html>"""
    
    return enhanced_html

def generate_shap_summary(shap_values, X_data, expected_value, output_dir):
    """
    Generate a summary of SHAP analysis across all customers
    """
    try:
        # Calculate feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        feature_names = X_data.columns
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Save summary statistics
        summary_file = Path(output_dir) / 'shap_summary.csv'
        importance_df.to_csv(summary_file, index=False)
        
        print(f"✅ SHAP summary saved to: {summary_file}")
        print(f"🏆 Top 5 Most Important Features:")
        for i, row in importance_df.head().iterrows():
            print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")
            
    except Exception as e:
        print(f"⚠️ Error generating summary: {e}")

def main():
    """
    Main function to orchestrate SHAP generation
    """
    print("🎯 SHAP Full Dataset Analysis")
    print("=" * 50)
    
    # Load model and data
    model, scaler, X_data = load_xgboost_model_and_data()
    
    if model is None or X_data is None:
        print("❌ Failed to load model or data. Please check file paths.")
        return
    
    # Ask user for configuration
    try:
        max_customers = input("\\nEnter max customers to analyze (or press Enter for all): ").strip()
        max_customers = int(max_customers) if max_customers else None
    except ValueError:
        max_customers = None
    
    # Generate SHAP plots
    success = generate_full_dataset_shap(
        model=model,
        X_data=X_data,
        max_customers=max_customers,
        output_dir='force_plots'
    )
    
    if success:
        print("\\n🎉 SHAP analysis complete!")
        print("\\n📋 Next Steps:")
        print("1. Start your Flask app: python app.py")
        print("2. Visit: http://127.0.0.1:5000/shap")
        print("3. Explore individual customer SHAP analyses")
        print("4. Check force_plots/ directory for HTML files")
        print("5. Review shap_summary.csv for feature importance")
    else:
        print("\\n❌ SHAP analysis failed. Check error messages above.")

if __name__ == "__main__":
    main()