# Data Analysis Platform

A clean and simple Flask web application for uploading CSV files and displaying churn analysis results.

## Features

- **File Upload**: Drag & drop or browse to upload CSV.
- **Explainability**: View comprehensive analysis results with:
  - Model performance metrics
  - Feature importance visualization
  - Predictions summary with charts
  - AI insights and recommendations
- **Clean UI**: Modern, responsive design with Bootstrap.
- **Interactive Charts**: Visual representations using Chart.js
- **Customer Churn Prediction Model**: Machine learning models to predict customer churn.
- **SHAP (SHapley Additive exPlanations)**: Unified approach to explain model predictions.
- **Feature Importance Analysis**: Understand which features matter most.
- **Interactive Visualizations**: Visual explanations of model behavior

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the application:

```bash
python app.py
```

3. Open your browser and navigate to `http://localhost:5000`

## Usage

1. **Upload Page**: Select or drag & drop a CSV.
2. **Analysis Page**: View AI predictions and insights based on your data
3. **Reset**: Start over with a new file

## File Structure

```
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── uploads/              # Uploaded files directory
├── static/
│   ├── style.css         # Custom CSS styles
│   └── script.js         # JavaScript functionality
└── templates/
    ├── index.html        # File upload page
    └── predictions.html  # AI predictions dashboard
```

## Technologies Used

- **Backend**: Flask, Pandas
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Charts**: Chart.js
- **Icons**: Font Awesome
- **MLModels**: XGBoost , MLP
-
-
## AI Chat links:
- https://chatgpt.com/share/68eaeb01-2f78-800a-bc1e-21872791b872
- https://chatgpt.com/share/68ea6377-fb98-8000-a7fd-e723171f20fb
- https://g.co/gemini/share/2658c751415e
