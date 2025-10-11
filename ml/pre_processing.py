# import re
# import os
# import numpy as np
# import pandas as pd
# import joblib
# from sklearn.preprocessing import OrdinalEncoder, StandardScaler



# def convert_range_to_midpoint(value):
#     """
#     Cleans a string, converts ranges to their midpoint, and handles various formats.
#     """
#     # Ensure value is a string and convert to lowercase
#     s = str(value).lower()
    
#     # Clean the string: remove $, ,, +, and replace 'k' with '000'
#     s = s.replace('$', '').replace(',', '').replace('+', '')
#     s = re.sub(r'k', '000', s)

#     # Check if it's a range (contains a hyphen)
#     if '-' in s:
#         try:
#             parts = s.split('-')
#             start = float(parts[0].strip())
#             end = float(parts[1].strip())
#             return (start + end) / 2
#         except (ValueError, IndexError):
#             return 0 # Return 0 if range is malformed
    
#     # If not a range, try converting to a single number
#     try:
#         return float(s.strip())
#     except ValueError:
#         return 0 # Return 0 if it's a non-numeric string


# def preprocessing_data(df , drop_cols = [] , categorical_encoder='ordinal' , path= 'data'):
#     print("Preprocessing data...")

#     # Combine all columns to be dropped into a single list
#     all_cols_to_drop = [col for col in df.columns if col.lower().endswith('_id')]
#     if 'Id' in df.columns and 'Id' not in all_cols_to_drop:
#         all_cols_to_drop.append('Id')
#     all_cols_to_drop.extend(drop_cols) # Add user-specified columns

#     # Find which of these columns actually exist in the DataFrame
#     existing_cols_to_drop = [col for col in all_cols_to_drop if col in df.columns]

#     if existing_cols_to_drop:
#         print(f"Dropping the following columns: {existing_cols_to_drop}")
#         df = df.drop(columns=existing_cols_to_drop)
#     else:
#         print("No specified columns to drop were found in the DataFrame.")


#     if "home_market_value" in df.columns:
#         # Set market value to 0 for non-homeowners
#         if 'home_owner' in df.columns:
#             df.loc[df['home_owner'] == 0, 'home_market_value'] = '0'
        
#         # FIX: Use reassignment instead of inplace=True
#         df['home_market_value'] = df['home_market_value'].fillna('0')

#         # Apply the conversion function
#         df['home_market_value'] = df['home_market_value'].apply(convert_range_to_midpoint)
    
#     df = df.dropna()
    
#     # For this version, we'll treat all remaining columns as features
#     # since no target column is specified
#     X = df.copy()

#     # Data Imputation
#     # Removing if the rows have any missing values or nan

#     # Identify categorical columns for encoding
#     categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
#     # Apply chosen encoding for categorical features
#     if categorical_encoder == 'onehot':
#         print(f"Applying One-Hot Encoding to: {list(categorical_cols)}")
#         if not categorical_cols.empty:
#             X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
#         else:
#             print("No categorical columns found for one-hot encoding.")
#     elif categorical_encoder == 'ordinal':
#         print(f"Applying Ordinal Encoding to: {list(categorical_cols)}")
#         if not categorical_cols.empty:
#             encoder = OrdinalEncoder()
#             X[categorical_cols] = encoder.fit_transform(X[categorical_cols])
#         else:
#             print("No categorical columns found for ordinal encoding.")
#     else:
#         raise ValueError("Invalid categorical_encoder. Choose 'onehot' or 'ordinal'.")


#     # Scale ALL numerical features using pre-trained scaler
#     print("Loading pre-trained StandardScaler parameters...")
#     scaler_path = os.path.join('ml', 'scaler_params.pkl')
    
#     # Load the pre-trained scaler using joblib
#     scaler = joblib.load(scaler_path)
#     print("Successfully loaded pre-trained scaler parameters.")
    
#     # Apply scaling to numerical columns
#     numerical_cols = X.select_dtypes(include=np.number).columns
#     if not numerical_cols.empty:
#         print(f"Applying scaling to columns: {list(numerical_cols)}")
#         X[numerical_cols] = scaler.transform(X[numerical_cols])
#     else:
#         print("No numerical columns found to scale.")

#     filepath = os.path.join(path, 'x_processed.csv')
#     print("Preprocessing complete.")
#     X.to_csv(filepath, index=False)
#     return X


import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def convert_range_to_midpoint(value):
    s = str(value).lower().replace('$', '').replace(',', '').replace('+', '')
    s = re.sub(r'k', '000', s)
    if '-' in s:
        try:
            parts = s.split('-')
            start = float(parts[0].strip())
            end = float(parts[1].strip())
            return (start + end) / 2
        except:
            return 0
    try:
        return float(s.strip())
    except:
        return 0

# Fixed feature list (105 features)
EXPECTED_FEATURES = [
    'curr_ann_amt','days_tenure','age_in_years','income','has_children',
    'length_of_residence','home_market_value','home_owner','college_degree',
    'good_credit','city_Aledo','city_Allen','city_Anna','city_Argyle','city_Arlington',
    'city_Aubrey','city_Azle','city_Balch Springs','city_Bedford','city_Blue Ridge',
    'city_Burleson','city_Caddo Mills','city_Carrollton','city_Cedar Hill','city_Celina',
    'city_Chatfield','city_Colleyville','city_Coppell','city_Crandall','city_Crowley',
    'city_Dallas','city_Denton','city_Desoto','city_Duncanville','city_Ennis','city_Era',
    'city_Euless','city_Farmersville','city_Ferris','city_Flower Mound','city_Forney',
    'city_Forreston','city_Fort Worth','city_Frisco','city_Garland','city_Grand Prairie',
    'city_Grapevine','city_Haltom City','city_Haslet','city_Hurst','city_Hutchins',
    'city_Irving','city_Italy','city_Joshua','city_Justin','city_Kaufman','city_Keller',
    'city_Kemp','city_Kennedale','city_Krum','city_Lake Dallas','city_Lancaster',
    'city_Lavon','city_Lewisville','city_Little Elm','city_Mansfield','city_Maypearl',
    'city_Mckinney','city_Melissa','city_Mertens','city_Mesquite','city_Midlothian',
    'city_Milford','city_Naval Air Station Jrb','city_Nevada','city_North Richland Hills',
    'city_Palmer','city_Pilot Point','city_Plano','city_Ponder','city_Princeton',
    'city_Prosper','city_Red Oak','city_Rice','city_Richardson','city_Roanoke',
    'city_Rockwall','city_Rowlett','city_Royse City','city_Sachse','city_Sanger',
    'city_Scurry','city_Seagoville','city_Southlake','city_Springtown','city_Sunnyvale',
    'city_Terrell','city_The Colony','city_Tioga','city_Valley View','city_Waxahachie',
    'city_Weatherford','city_Wilmer','city_Wylie','marital_status_Single'
]

def preprocessing_data(df, drop_cols=[], categorical_encoder='ordinal', path='data'):
    print("Preprocessing data...")

    # Drop columns
    all_cols_to_drop = [col for col in df.columns if col.lower().endswith('_id')]
    if 'Id' in df.columns and 'Id' not in all_cols_to_drop:
        all_cols_to_drop.append('Id')
    all_cols_to_drop.extend(drop_cols)
    df = df.drop(columns=[c for c in all_cols_to_drop if c in df.columns], errors='ignore')

    # Handle home_market_value
    if "home_market_value" in df.columns:
        if 'home_owner' in df.columns:
            df.loc[df['home_owner'] == 0, 'home_market_value'] = '0'
        df['home_market_value'] = df['home_market_value'].fillna('0').apply(convert_range_to_midpoint)

    df = df.dropna()

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if categorical_encoder == 'onehot':
        print(f"Applying One-Hot Encoding to: {list(categorical_cols)}")
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    elif categorical_encoder == 'ordinal':
        print(f"Applying Ordinal Encoding to: {list(categorical_cols)}")
        if len(categorical_cols) > 0:
            encoder = OrdinalEncoder()
            df[categorical_cols] = encoder.fit_transform(df[categorical_cols])
    else:
        raise ValueError("Invalid categorical_encoder. Choose 'onehot' or 'ordinal'.")

    # Load pre-trained scaler
    print("Loading pre-trained StandardScaler parameters...")
    scaler_path = os.path.join('ml', 'scaler_params.pkl')
    scaler = joblib.load(scaler_path)
    print("Successfully loaded pre-trained scaler parameters.")

    numerical_cols = df.select_dtypes(include=np.number).columns
    if len(numerical_cols) > 0:
        df[numerical_cols] = scaler.transform(df[numerical_cols])

    # ✅ Ensure all expected features exist
    for feature in EXPECTED_FEATURES:
        if feature not in df.columns:
            df[feature] = 0

    # ✅ Keep only expected features, in correct order
    df = df[EXPECTED_FEATURES]

    # Save
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, 'x_processed.csv')
    df.to_csv(filepath, index=False)
    print(f"Preprocessing complete. Saved to {filepath}")
    return df
