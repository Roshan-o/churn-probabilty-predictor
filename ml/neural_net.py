# -------------------------------
# IMPORTS
# -------------------------------
import pandas as pd
import numpy as np
import re
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import joblib

# -------------------------------
# PREPROCESSING UTILITY
# -------------------------------
def convert_range_to_midpoint(value):
    s = str(value).lower().replace('$', '').replace(',', '').replace('+', '')
    s = re.sub(r'k', '000', s)
    if '-' in s:
        try:
            parts = s.split('-')
            return (float(parts[0].strip()) + float(parts[1].strip())) / 2
        except (ValueError, IndexError):
            return 0
    try:
        return float(s.strip())
    except ValueError:
        return 0

# -------------------------------
# HYPERPARAMETERS & CONFIG
# -------------------------------
SEED = 42
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2  # L2 regularization
DROPOUT_RATE = 0.3   # Dropout regularization
EPOCHS = 100
BATCH_SIZE = 256
EARLY_STOP_PATIENCE = 10

# -------------------------------
# REPRODUCIBILITY
# -------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# # -------------------------------
# # STEP 1: LOAD AND CLEAN DATA (WITHOUT SCALING)
# # -------------------------------
# print("🔧 Starting Data Loading and Cleaning...")
# df = pd.read_csv("autoinsurance_churn.csv")

# # Drop specified and ID columns
# drop_cols = ["latitude", "longitude", "county", "state", "cust_orig_date", "date_of_birth", "acct_suspd_date"]
# all_cols_to_drop = [col for col in df.columns if col.lower().endswith('_id')]
# if 'Id' in df.columns and 'Id' not in all_cols_to_drop:
#     all_cols_to_drop.append('Id')
# all_cols_to_drop.extend(drop_cols)
# existing_cols_to_drop = [col for col in all_cols_to_drop if col in df.columns]
# df = df.drop(columns=existing_cols_to_drop)
# print(f"Dropped columns: {existing_cols_to_drop}")

# # Handle home_market_value
# if "home_market_value" in df.columns:
#     if 'home_owner' in df.columns:
#         df.loc[df['home_owner'] == 0, 'home_market_value'] = '0'
#     df['home_market_value'] = df['home_market_value'].fillna('0')
#     df['home_market_value'] = df['home_market_value'].apply(convert_range_to_midpoint)

# df = df.dropna()
# X = df.drop(columns=['Churn'])
# y = df['Churn']
# categorical_cols = X.select_dtypes(include=['object', 'category']).columns
# print(f"Applying One-Hot Encoding to: {list(categorical_cols)}")
# X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
# print("✅ Data Cleaning and Encoding complete.")

# # -------------------------------
# # STEP 2: SPLIT THE DATA (BEFORE SCALING)
# # -------------------------------
# print("🔪 Splitting data into train, validation, and test sets...")
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=SEED, stratify=y_train)
# print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

# # -------------------------------
# # STEP 3: SCALE THE DATA (CORRECT WORKFLOW)
# # -------------------------------
# print("📏 Scaling numerical features...")
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_val_scaled = scaler.transform(X_val)
# X_test_scaled = scaler.transform(X_test)
# print("✅ Scaler fitted on training data.")

# # -------------------------------
# # STEP 4: PYTORCH DATA SETUP
# # -------------------------------
# y_train = y_train.values.astype(np.float32)
# y_val   = y_val.values.astype(np.float32)
# y_test  = y_test.values.astype(np.float32)

class TabularDataset(Dataset):
    def _init_(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    def _len_(self):
        return len(self.y)
    def _getitem_(self, idx):
        return self.X[idx], self.y[idx]

# train_loader = DataLoader(TabularDataset(X_train_scaled, y_train), batch_size=BATCH_SIZE, shuffle=True)
# val_loader   = DataLoader(TabularDataset(X_val_scaled, y_val), batch_size=BATCH_SIZE, shuffle=False)
# test_loader  = DataLoader(TabularDataset(X_test_scaled, y_test), batch_size=BATCH_SIZE, shuffle=False)

# -------------------------------
# STEP 5: PYTORCH MODEL, LOSS, AND OPTIMIZER
# -------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        d_in = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d_in, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(p=dropout))
            d_in = h
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# # Use hyperparameters from the config section
# model = MLP(input_dim=X_train_scaled.shape[1], hidden_dims=[256, 128, 64], dropout=DROPOUT_RATE).to(device)
# criterion = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

# -------------------------------
# STEP 6: MODEL TRAINING AND EVALUATION
# -------------------------------
def evaluate(model, loader, criterion, device):
    model.eval()
    ys, probs = [], []
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            prob = torch.sigmoid(logits).cpu().numpy().flatten()
            probs.append(prob)
            ys.append(yb.cpu().numpy().flatten())
            loss_batch = criterion(logits, yb).item()
            losses.append(loss_batch)
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(probs)
    avg_loss = np.mean(losses)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    y_pred_label = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred_label)
    prec = precision_score(y_true, y_pred_label, zero_division=0)
    rec = recall_score(y_true, y_pred_label, zero_division=0)
    return {"loss": avg_loss, "auc": auc, "acc": acc, "prec": prec, "rec": rec}

# best_val_loss = float('inf')
# epochs_no_improve = 0
# best_state = None

# print("\n🚀 Starting model training...")
# for epoch in range(1, EPOCHS + 1):
#     model.train()
#     epoch_losses = []
#     for xb, yb in train_loader:
#         xb, yb = xb.to(device), yb.to(device)
#         optimizer.zero_grad()
#         logits = model(xb)
#         loss = criterion(logits, yb)
#         loss.backward()
#         optimizer.step()
#         epoch_losses.append(loss.item())
    
#     train_loss = np.mean(epoch_losses)
#     val_metrics = evaluate(model, val_loader, criterion, device)
#     val_loss = val_metrics["loss"]
#     scheduler.step(val_loss)
    
#     print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_metrics['auc']:.4f}")
    
#     if val_loss < best_val_loss - 1e-6:
#         best_val_loss = val_loss
#         epochs_no_improve = 0
#         best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
#     else:
#         epochs_no_improve += 1
#         if epochs_no_improve >= EARLY_STOP_PATIENCE:
#             print(f"🛑 Early stopping triggered after {epoch} epochs.")
#             break

# if best_state is not None:
#     print("\nLoading best model state for final evaluation.")
#     model.load_state_dict(best_state)
#     model.to(device)

# test_metrics = evaluate(model, test_loader, criterion, device)
# print("\n--- Test Set Results ---")
# print(f"Test Loss: {test_metrics['loss']:.4f}")
# print(f"Test AUC:  {test_metrics['auc']:.4f}")
# print(f"Test Acc:  {test_metrics['acc']:.4f} | Precision: {test_metrics['prec']:.4f} | Recall: {test_metrics['rec']:.4f}")

# -------------------------------
# STEP 7: PREDICTION FUNCTION & FINAL SAVING
# -------------------------------
def predict_proba_numpy(model, X_np, batch_size=1024):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, X_np.shape[0], batch_size):
            xb = torch.tensor(X_np[i:i+batch_size], dtype=torch.float32).to(device)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(probs).flatten()

# test_probs = predict_proba_numpy(model, X_test_scaled)
# pd.DataFrame({"y_true": y_test, "y_prob": test_probs}).to_csv("test_predictions.csv", index=False)
# print("\n✅ Test predictions saved to 'test_predictions.csv'.")

# print("💾 Saving final model and scaler...")
# # Save the model state dictionary (recommended PyTorch method)
# torch.save(model.state_dict(), "mlp_model.pth")
# print("✅ Model state dictionary saved to 'mlp_model.pth'.")

# # Save the full model object and scaler using joblib
# joblib.dump(model, "mlp_model.pkl")
# joblib.dump(scaler, "mlp_scaler.pkl")
# print("✅ Model object saved to 'mlp_model.pkl'.")
# print("✅ Scaler saved to 'mlp_scaler.pkl'.")


if __name__ == "__main__":
    pass