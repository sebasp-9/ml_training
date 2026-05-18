import numpy as np
import pandas as pd
from sklearn.metrics import  accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ==============================================================
# 1. LOAD DATA
# ==============================================================

TRAIN_DATA = 'KDDTrain+.txt'
TEST_DATA = 'KDDTest+.txt'

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'class', 'level'
]

print("Loading data...")
df_train = pd.read_csv(TRAIN_DATA, names=columns)
df_test = pd.read_csv(TEST_DATA, names=columns)

# Drop difficulty level column (not a feature)
df_train.drop(columns=['level'], inplace=True)
df_test.drop(columns=['level'], inplace=True)

print(f"Training set: {df_train.shape[0]} records, {df_train.shape[1]} columns")
print(f"Test set:     {df_test.shape[0]} records, {df_test.shape[1]} columns")

# ==============================================================
# 2. ENCODE CATEGORICAL FEATURES
# ==============================================================

# Merge temporarily to ensure consistent encoding across train and test
df_full = pd.concat([df_train, df_test])

# Encode categorical columns as integers
cat_cols = ['protocol_type', 'service', 'flag']
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df_full[col] = le.fit_transform(df_full[col])
    label_encoders[col] = le

# ==============================================================
# 3. MAP ATTACKS TO 5 CATEGORIES
# ==============================================================

category_map = {
    'normal': 'Normal',
    # DoS
    'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS',
    'processtable': 'DoS', 'udpstorm': 'DoS', 'worm': 'DoS',
    # Probe
    'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    # R2L
    'warezclient': 'R2L', 'guess_passwd': 'R2L', 'ftp_write': 'R2L',
    'imap': 'R2L', 'phf': 'R2L', 'multihop': 'R2L', 'warezmaster': 'R2L',
    'spy': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L',
    'snmpgetattack': 'R2L', 'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
    # U2R
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R',
    'perl': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R'
}

df_full['category'] = df_full['class'].map(category_map).fillna('Other')

# ==============================================================
# 4. PREPARE FEATURES AND LABELS
# ==============================================================

# Drop constant column and original class labels
df_full.drop(columns=['num_outbound_cmds', 'class'], inplace=True)

# Split back into train and test
train_len = len(df_train)
df_train_processed = df_full.iloc[:train_len].copy()
df_test_processed = df_full.iloc[train_len:].copy()

x_train = df_train_processed.drop(columns=['category'])
y_train = df_train_processed['category']

x_test = df_test_processed.drop(columns=['category'])
y_test = df_test_processed['category']

print(f"\nFeatures: {x_train.shape[1]}")
print(f"\nTraining set class distribution:")
print(y_train.value_counts())
print(f"\nTest set class distribution:")
print(y_test.value_counts())

# ==============================================================
# YOUR WORK STARTS HERE
# ==============================================================
# 
# You now have:
#   x_train, y_train  — training features and labels (5 categories)
#   x_test, y_test    — test features and labels (5 categories)
#
# Your task:
#   1. Train one or more models on x_train / y_train
#   2. Predict on x_test
#   3. Evaluate using macro F1-score
#
# Useful imports for evaluation:
#   from sklearn.metrics import classification_report, confusion_matrix, f1_score
#
# To compute macro F1:
#   f1_score(y_test, y_pred, average='macro')

"""
assignment also specifies scale_pos_weight, how to utilize?
- https://xgboost.readthedocs.io/en/stable/python/python_api.html
- https://xgboosting.com/xgboost-configure-xgboost.train-parameters/
- https://xgboosting.com/xgboost-configure-scale_pos_weight-parameter/
"""
model = XGBClassifier(
    n_estimators=200,                       # 🧮 TUNE HERE
    max_depth=6,                            # 🧮 TUNE HERE
    learning_rate=0.1,                      # 🧮 TUNE HERE
    subsample=0.8,                          # 🧮 TUNE HERE
    colsample_bytree=0.8,
    objective="multi:softprob",             # for multiclass, think this is required for the dataset
    num_class = len(np.unique(y_train)),    # specify number of multiclass for above
    eval_metric="logloss",                  # unsure on this parameter, maybe should be 'error'?
    random_state=42,                        # keep static for reproducibility, per assignment
    n_jobs=-1
)

# train
model.fit(
    x_train,
    y_train,
    eval_set=[(x_test, y_test)],
    verbose=True
)

# predict
y_pred = model.predict(x_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f1_score(y_test, y_pred, average='macro'))

# save model
#model.save_model("xgboost_intrusion_model.json")
