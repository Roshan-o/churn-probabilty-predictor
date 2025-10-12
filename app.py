from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_from_directory, render_template_string
import os
import pandas as pd
import numpy as np
import json 
from werkzeug.utils import secure_filename
from ml.pre_processing import preprocessing_data
# Import predict function to run model after preprocessing
from ml.predict import predict
# Import AI insight functions from new_gemini.py
from ml.new_gemini import task1_prepare_and_store_all_ml_data, task2_generate_strategy_on_demand
import shap
import joblib
from pathlib import Path

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'data'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create data directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variable to store prepared ML data
prepared_ml_data = None

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Homepage with file upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = 'x_test.csv'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Store filename in session for analysis page
        session['uploaded_file'] = filename
        
        # Basic file analysis
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
            
            # Store basic info in session
            session['file_info'] = {
                'filename': filename,
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns)
            }
            
            # Preprocess data
            processed_data = preprocessing_data(df , drop_cols=["latitude" , "longitude"  , "county" , "state" , "cust_orig_date" , "date_of_birth" , "acct_suspd_date" ] , categorical_encoder='onehot' , path= app.config['UPLOAD_FOLDER'])

            # Run model predictions (synchronously). This will read the processed file and produce
            # `data/x_predicted_output.csv` per ml/predict.py implementation.
            try:
                results = predict()
                # Store output file path for later pages
                # session['predicted_output'] = results.get('output_file') if isinstance(results, dict) else 'data/x_predicted_output.csv'
            except Exception as e:
                # If prediction fails, flash and redirect to index
                flash(f'Prediction failed: {str(e)}')
                return redirect(url_for('index'))

            return redirect(url_for('predictions'))
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload CSV or Excel files only.')
        return redirect(url_for('index'))

@app.route('/predictions')
def predictions():
    """AI predictions page with analysis results"""
    if 'file_info' not in session:
        flash('Please upload a file first')
        return redirect(url_for('index'))
    
    file_info = session.get('file_info')
    
    # Placeholder predictions data
    placeholder_predictions = {
        'accuracy_score': 92.5,
        'model_type': 'Random Forest Classifier',
        'feature_importance': [
            {'feature': 'Feature_1', 'importance': 0.35},
            {'feature': 'Feature_2', 'importance': 0.28},
            {'feature': 'Feature_3', 'importance': 0.22},
            {'feature': 'Feature_4', 'importance': 0.15}
        ],
        'predictions_summary': {
            'total_predictions': file_info['rows'] if file_info else 100,
            'positive_predictions': 67,
            'negative_predictions': 33
        },
        'model_metrics': {
            'precision': 0.91,
            'recall': 0.89,
            'f1_score': 0.90
        }
    }
    
    return render_template('predictions.html', 
                         file_info=file_info, 
                         predictions=placeholder_predictions)

@app.route('/reset')
def reset():
    """Reset session and go back to upload page"""
    session.clear()
    flash('Session reset. You can upload a new file.')
    return redirect(url_for('index'))



@app.route('/customer-analysis')
def customer_analysis():
    """Customer analysis page with individual predictions formatted for frontend"""
    processed_file = os.path.join(app.config['UPLOAD_FOLDER'], 'x_predicted_output.csv')
    
    if not os.path.exists(processed_file):
        flash('Processed file not found. Please upload and preprocess data first.')
        return redirect(url_for('index'))
    
    # Read processed data
    # df = pd.read_csv(processed_file)
    
    customers_data = []
    
    # Generate mock churn probabilities (you can replace this with actual model predictions)
    df = pd.read_csv(r'data\x_predicted_output.csv')
    
    for idx, row in df.iterrows():
        prob = row['final_prediction']
        churn_probability = round(prob * 100, 1)
        
        # Determine risk level
        if prob >= 0.7:
            risk_level = 'high'
            segment = 'Premium'
        elif prob >= 0.3:
            risk_level = 'medium'
            segment = 'Standard'
        else:
            risk_level = 'low'
            segment = 'Basic'
        
        # Example derived features
        tenure = int((row.get('days_tenure', 0) + 2) * 12)  # scaled mock months
        monthly_charges = abs(round(row.get('curr_ann_amt', 0) * 10 + 60, 2))
        total_charges = round(monthly_charges * (tenure / 12), 2)
        contract_type = np.random.choice(['Month-to-Month', '1 Year', '2 Year'])
        payment_method = np.random.choice(['Electronic Check', 'Credit Card', 'Bank Transfer'])
        internet_service = np.random.choice(['Fiber Optic', 'DSL', 'None'])
        
        # Feature summary list
        features = [
            f"Tenure: {tenure} months",
            f"Monthly Charges: ${monthly_charges}",
            f"Total Charges: ${total_charges}",
            f"Contract: {contract_type}",
            f"Payment: {payment_method}",
            f"Internet: {internet_service}"
        ]
        
        # Random mock feature importance
        feature_importance = np.random.randint(40, 100, 5).tolist()
        
        # Dynamic recommendations
        if risk_level == 'high':
            recommendations = [
                'Offer contract upgrade with 20% discount',
                'Suggest automatic payment setup',
                'Provide personalized service benefits',
                'Schedule quarterly review calls'
            ]
        elif risk_level == 'medium':
            recommendations = [
                'Send satisfaction survey',
                'Provide loyalty reward points',
                'Offer seasonal plan discounts',
                'Monitor churn indicators closely'
            ]
        else:
            recommendations = [
                'Encourage referral program participation',
                'Promote value-added services',
                'Maintain regular engagement',
                'Offer thank-you rewards'
            ]
        
        # Create customer dictionary
        customer = {
            'id': row['individual_id'],
            'name': f'Customer {idx + 1}',
            'segment': segment,
            'risk_level': risk_level,
            'churn_probability': churn_probability,
            # expose raw values from processed output so frontend can display them
            'confidence_score': row.get('confidence_score'),
            'age_in_years': row.get('age_in_years'),
            'curr_ann_amt': row.get('curr_ann_amt'),
            'tenure': tenure,
            'monthly_charges': monthly_charges,
            'total_charges': total_charges,
            'contract_type': contract_type,
            'payment_method': payment_method,
            'internet_service': internet_service,
            'features': features,
            'feature_importance': feature_importance,
            'recommendations': recommendations
        }
        
        customers_data.append(customer)
    
    # Convert to JSON for frontend JavaScript
    customers_json = json.dumps(customers_data)
    
    return render_template('customer_analysis.html', customers_json=customers_json)

@app.route('/generate_ai_insight/<int:customer_index>')
def generate_ai_insight(customer_index):
    """Generate AI insight for a specific customer using the imported functions"""
    global prepared_ml_data
    
    try:
        # Initialize ML data if not already done
        if prepared_ml_data is None:
            print("Preparing ML data for the first time...")
            prepared_ml_data = task1_prepare_and_store_all_ml_data()
            if prepared_ml_data is None:
                return jsonify({'error': 'Failed to prepare ML data', 'success': False})
        
        # Generate strategy for the specific customer
        strategy = task2_generate_strategy_on_demand(customer_index, prepared_ml_data)
        
        # Check if strategy generation was successful
        if strategy.startswith("ERROR:") or strategy.startswith("Error:"):
            return jsonify({'error': strategy, 'success': False})
        
        return jsonify({
            'success': True,
            'insight': strategy,
            'customer_index': customer_index
        })
        
    except Exception as e:
        print(f"Error generating AI insight for customer {customer_index}: {e}")
        return jsonify({
            'error': f'Failed to generate AI insight: {str(e)}',
            'success': False
        })

@app.route('/shap')
def shap_home():
    """SHAP visualizations home page"""
    # Get list of available SHAP plots
    shap_plots_dir = Path('force_plots')
    shap_files = []
    
    if shap_plots_dir.exists():
        for file in shap_plots_dir.glob('force_plot_customer_*.html'):
            customer_id = file.stem.replace('force_plot_customer_', '')
            shap_files.append({
                'customer_id': customer_id,
                'filename': file.name,
                'title': f'Customer {customer_id} SHAP Analysis'
            })
    
    # Sort by customer ID numerically
    shap_files.sort(key=lambda x: int(x['customer_id']))
    
    return render_template('shap_home.html', shap_files=shap_files)

@app.route('/shap/customer/<int:customer_id>')
def shap_customer(customer_id):
    """Display SHAP visualization for a specific customer"""
    file_path = Path(f'force_plots/force_plot_customer_{customer_id}.html')
    
    if not file_path.exists():
        flash(f'SHAP visualization for customer {customer_id} not found.')
        return redirect(url_for('shap_home'))
    
    # Read the HTML content
    with open(file_path, 'r', encoding='utf-8') as f:
        shap_html = f.read()
    
    # Enhanced HTML template with SHAP JavaScript library
    enhanced_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SHAP Analysis - Customer {customer_id}</title>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background-color: #2c3e50;
                color: white;
                padding: 20px;
                margin: -20px -20px 20px -20px;
                text-align: center;
            }}
            .back-button {{
                display: inline-block;
                margin-bottom: 20px;
                padding: 10px 20px;
                background-color: #3498db;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                transition: background-color 0.3s;
            }}
            .back-button:hover {{
                background-color: #2980b9;
            }}
            .shap-container {{
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }}
            .explanation {{
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
                border-left: 4px solid #3498db;
            }}
        </style>
        <script src="https://cdn.jsdelivr.net/gh/slundberg/shap@master/js/dist/bundle.js"></script>
    </head>
    <body>
        <div class="header">
            <h1>SHAP Force Plot Analysis</h1>
            <h2>Customer {customer_id}</h2>
        </div>
        
        <a href="/shap" class="back-button">← Back to SHAP Home</a>
        
        <div class="explanation">
            <h3>Understanding SHAP Force Plots</h3>
            <p>This visualization shows how each feature contributes to the model's prediction for this customer:</p>
            <ul>
                <li><strong>Red features</strong> push the prediction towards higher churn probability</li>
                <li><strong>Blue features</strong> push the prediction towards lower churn probability</li>
                <li>The width of each feature bar represents the magnitude of its impact</li>
                <li>The base value represents the average prediction across all customers</li>
            </ul>
        </div>
        
        <div class="shap-container">
            {shap_html.replace('<html>', '').replace('</html>', '').replace('<head>', '').replace('</head>', '').replace('<body>', '').replace('</body>', '')}
        </div>
        
        <div class="explanation">
            <h3>Next Steps</h3>
            <p>Based on this SHAP analysis, you can:</p>
            <ul>
                <li>Focus retention efforts on the red (risk-increasing) features</li>
                <li>Leverage the blue (protective) features in your strategy</li>
                <li>Compare this analysis with other customers to identify patterns</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    return render_template_string(enhanced_html)

@app.route('/shap/generate')
def generate_shap():
    """Generate new SHAP visualizations (example endpoint)"""
    try:
        flash('SHAP visualizations would be generated here. Implement with your actual model and data.')
        return redirect(url_for('shap_home'))
        
    except Exception as e:
        flash(f'Error generating SHAP visualizations: {str(e)}')
        return redirect(url_for('shap_home'))

@app.route('/shap/files/<filename>')
def serve_shap_file(filename):
    """Serve SHAP HTML files directly"""
    return send_from_directory('force_plots', filename)

@app.route('/shap/summary')
def shap_summary():
    """Display SHAP summary visualization"""
    summary_file_path = Path('shap_plots/force_plots/summary.html')
    
    if not summary_file_path.exists():
        flash('SHAP summary visualization not found.')
        return redirect(url_for('predictions'))
    
    # Read the HTML content
    with open(summary_file_path, 'r', encoding='utf-8') as f:
        summary_html = f.read()
    
    # Enhanced HTML template for summary
    enhanced_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SHAP Summary Analysis</title>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background-color: #2c3e50;
                color: white;
                padding: 20px;
                margin: -20px -20px 20px -20px;
                text-align: center;
            }}
            .back-button {{
                display: inline-block;
                margin-bottom: 20px;
                padding: 10px 20px;
                background-color: #3498db;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                transition: background-color 0.3s;
            }}
            .back-button:hover {{
                background-color: #2980b9;
            }}
            .shap-container {{
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }}
            .explanation {{
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
                border-left: 4px solid #3498db;
            }}
        </style>
        <script src="https://cdn.jsdelivr.net/gh/slundberg/shap@master/js/dist/bundle.js"></script>
    </head>
    <body>
        <div class="header">
            <h1>SHAP Summary Analysis</h1>
            <h2>Overall Model Insights</h2>
        </div>
        
        <a href="/predictions" class="back-button">← Back to Dashboard</a>
        
        <div class="explanation">
            <h3>Understanding SHAP Summary Plots</h3>
            <p>This visualization provides an overview of feature importance across all customers:</p>
            <ul>
                <li><strong>Feature ranking</strong> shows which features are most important globally</li>
                <li><strong>Distribution patterns</strong> reveal how feature effects vary across customers</li>
                <li><strong>Color coding</strong> indicates feature values (red = high, blue = low)</li>
                <li><strong>Spread</strong> shows the range of SHAP values for each feature</li>
            </ul>
        </div>
        
        <div class="shap-container">
            {summary_html.replace('<html>', '').replace('</html>', '').replace('<head>', '').replace('</head>', '').replace('<body>', '').replace('</body>', '')}
        </div>
        
        <div class="explanation">
            <h3>Key Insights</h3>
            <p>Use this summary analysis to:</p>
            <ul>
                <li>Identify the most influential features for churn prediction</li>
                <li>Understand the typical direction of feature effects</li>
                <li>Spot features with high variability in their impact</li>
                <li>Guide feature engineering and data collection priorities</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    return render_template_string(enhanced_html)

if __name__ == '__main__':
    app.run(debug=True)
