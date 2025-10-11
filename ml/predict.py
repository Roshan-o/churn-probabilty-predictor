import xgboost as xgb
import joblib
import os
import pandas as pd
import numpy as np
try:
    # Use package-relative import when imported as part of the `ml` package
    from .neural_net import MLP
except Exception:
    # Fallback for running the script directly
    from neural_net import MLP
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# -------------------------------
# REQUIRED UTILITIES (Add these to predict.py)
# -------------------------------
def predict_proba_numpy(model, X_np, device, batch_size=1024):
    """
    Makes probability predictions using the PyTorch model on a numpy array.
    """
    model.eval() # Set model to evaluation mode
    probs = []
    with torch.no_grad():
        for i in range(0, X_np.shape[0], batch_size):
            xb = torch.tensor(X_np[i:i+batch_size], dtype=torch.float32).to(device)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(probs).flatten()

def combined_confidence(p1, p2, w_agree=0.4, w_cert=0.6):
    """
    Combines two model churn probabilities into a single confidence score.
    p1, p2: arrays or floats (probabilities)
    Returns: confidence score (same shape as inputs)
    """
    p1 = np.array(p1)
    p2 = np.array(p2)

    # 1. Agreement term
    agreement = 1 - np.abs(p1 - p2)

    # 2. Certainty term
    p_avg = (p1 + p2) / 2
    c = 4 * (p_avg - 0.5)**2

    # 3. Weighted combination
    confidence = w_agree * agreement + w_cert * c

    # Clip to [0,1]
    confidence = np.clip(confidence, 0, 1)
    p_final = np.maximum(p1, p2)
    return confidence, p_final

def predict(path="data/x_processed.csv"):
    print(f"Loading data from: {path}")
    # Read original x_test early so we can attempt ID-based alignment
    df_features_x = pd.read_csv(r'data\x_test.csv')
    df_features = pd.read_csv(path)
    X_np = df_features.values.astype(np.float32)
    # ----------------------------------------------------
    # 1. XGBoost Prediction
    # ----------------------------------------------------
    print("Loading XGBoost model...")
    model_xgb = xgb.XGBClassifier()
    # Load the XGBoost model from the specified path
    model_xgb.load_model(os.path.join('ml', 'xgb_churn_model.json'))
    # XGBoost prediction (using the potentially scaled data)
    # Note: If XGBoost was trained on *unscaled* data, this prediction will be 
    # slightly inconsistent/less optimal since we are feeding it *scaled* data.
    predictions_xgb = model_xgb.predict_proba(X_np)[:, 1]  # Get probability for class 1
    
    # # Load MLP model and make predictions
    print("Loading MLP model...")

    DROPOUT_RATE = 0.3
    INPUT_DIM = df_features.shape[1]
    HIDDEN_DIMS = [256,128,64]
    model_mlp = MLP(input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT_RATE).to(device)

    model_mlp.load_state_dict(torch.load(os.path.join('ml', 'mlp_model.pth'), map_location=device))

    print("Making MLP predictions...")
    BATCH_SIZE = 256
    predictions_mlp = predict_proba_numpy(model_mlp, X_np, device, batch_size=BATCH_SIZE)

    # Calculate confidence scores
    print("Calculating confidence scores...")
    confidence_scores, final_predictions = combined_confidence(predictions_xgb, predictions_mlp)

    # Prepare predictions dataframe
    pred_df = pd.DataFrame({
        'predictions_xgb': predictions_xgb,
        'predictions_mlp': predictions_mlp,
        'confidence_score': confidence_scores,
        'final_prediction': final_predictions
    })

    # Try to align by a stable identifier if available in both dataframes
    id_candidates = ['individual_id', 'address_id', 'id', 'customer_id']
    id_col = None
    for c in id_candidates:
        if c in df_features.columns and c in df_features_x.columns:
            id_col = c
            break

    if id_col is not None:
        # Use the id column from the processed features as index for predictions
        print(f"Aligning predictions to x_test by id column: {id_col}")
        # build pred index -> ensure lengths match by trimming preds to df_features length
        n = min(len(df_features), len(pred_df))
        pred_df = pred_df.iloc[:n].copy()
        pred_df.index = df_features.iloc[:n][id_col].values

        # set index on x_test and join (left join to preserve all x_test rows)
        out_indexed = df_features_x.set_index(id_col)
        out_indexed = out_indexed.join(pred_df, how='left')
        output_df = out_indexed.reset_index()
    else:
        # No id column available in both -> align by position but don't assume equal lengths
        print("No shared ID column found; aligning by position and filling missing values with NaN")
        output_df = df_features_x.copy()
        n_pred = len(pred_df)
        # assign for the overlapping prefix
        overlap = min(len(output_df), n_pred)
        for col in pred_df.columns:
            output_df.loc[:overlap-1, col] = pred_df[col].iloc[:overlap].values
        # leave remaining rows as NaN for prediction columns

    # Save the updated file
    # Ensure the output includes the original x_test columns
    output_path = "data/x_predicted_output.csv"
    # If df_features_x has an index column (from CSV), drop it
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")

    # Print first 10 predictions
    print("\n=== First 10 Predictions ===")
    for i in range(min(10, len(predictions_xgb))):
        print(f"Row {i+1}: XGB={predictions_xgb[i]:.4f}, MLP={predictions_mlp[i]:.4f}, "
              f"Confidence={confidence_scores[i]:.4f}, Final_P={final_predictions[i]:.4f}")
    
    return {
        'predictions_xgb': predictions_xgb,
        'predictions_mlp': predictions_mlp,
        'confidence_scores': confidence_scores,
        'final_predictions': final_predictions,
        'output_file': output_path
    }


if __name__ == "__main__":
    # Run predictions
    results = predict()
    print("\nPrediction process completed!")
