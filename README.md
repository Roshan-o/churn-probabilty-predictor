# Data Analysis Platform

A clean and simple Flask web application for uploading CSV/Excel files and displaying AI-powered analysis results.

## Features

- **File Upload**: Drag & drop or browse to upload CSV/Excel files
- **AI Predictions**: View comprehensive analysis results with:
  - Model performance metrics
  - Feature importance visualization
  - Predictions summary with charts
  - AI insights and recommendations
- **Clean UI**: Modern, responsive design with Bootstrap
- **Interactive Charts**: Visual representations using Chart.js

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

1. **Upload Page**: Select or drag & drop a CSV/Excel file
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

## Note

This application currently uses placeholder data for AI predictions. In a production environment, you would integrate with actual machine learning models and data processing pipelines.
