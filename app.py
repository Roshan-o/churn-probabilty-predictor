from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
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

if __name__ == '__main__':
    app.run(debug=True)
