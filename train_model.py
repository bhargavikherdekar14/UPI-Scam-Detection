import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import pickle

# Load the dataset
df = pd.read_csv('upi_scam_dataset.csv')

# Fill missing values
df['transaction_amount'].fillna(df['transaction_amount'].median(), inplace=True)

# Feature engineering
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek

# Drop unnecessary columns
df.drop(columns=['upi_holder_name', 'dob', 'datetime', 'upi_number'], inplace=True)

# Label Encoding
state_encoder = LabelEncoder()
seller_encoder = LabelEncoder()
merchant_encoder = LabelEncoder()

df['state'] = state_encoder.fit_transform(df['state'])
df['seller_name'] = seller_encoder.fit_transform(df['seller_name'])
df['merchant_category'] = merchant_encoder.fit_transform(df['merchant_category'])

# Separate features and target
X = df.drop('is_scam', axis=1)
y = df['is_scam']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Random Forest Classifier with GridSearchCV
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Best model
best_rf = grid_search.best_estimator_

# Evaluate the model
y_pred = best_rf.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Only keep scam transactions (label = 1)
scam_transactions = df[df['is_scam'] == 1]

# Save the model and encoders
with open('model.pkl', 'wb') as f:
    pickle.dump(best_rf, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('state_encoder.pkl', 'wb') as f:
    pickle.dump(state_encoder.classes_, f)

with open('seller_encoder.pkl', 'wb') as f:
    pickle.dump(seller_encoder.classes_, f)

with open('merchant_encoder.pkl', 'wb') as f:
    pickle.dump(merchant_encoder.classes_, f)
