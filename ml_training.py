import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import  accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


def create_confusion_matrix(y_test, y_pred):
    labels = [0, 1, 2, 3, 4]
    display_labels = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=display_labels, yticklabels=display_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    # plt.show()

parser = argparse.ArgumentParser(prog='llm_training', description="implements XGBoost to train network intrusion detection")
parser.add_argument('-n_estimators', '-n', default=100, help="number of trees in model")
parser.add_argument('-max_depth', '-d', default=6, help="maximum depth of model trees")
parser.add_argument('-learning_rate', '-l', default=0.3, help="(0-1) step size of each boost iteration")
parser.add_argument('-subsample', '-s', default=1, help="(0-1) fraction of observation on each tree")
args = parser.parse_args()

# load data -----------------------------------------------------------------------------------------------------------
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

print("🚚 Loading data...")
df_train = pd.read_csv(TRAIN_DATA, names=columns)
df_test = pd.read_csv(TEST_DATA, names=columns)

# Drop difficulty level column (not a feature)
df_train.drop(columns=['level'], inplace=True)
df_test.drop(columns=['level'], inplace=True)

#print(f"Training set: {df_train.shape[0]} records, {df_train.shape[1]} columns")
#print(f"Test set:     {df_test.shape[0]} records, {df_test.shape[1]} columns")

# encode categorical features -----------------------------------------------------------------------------------------
# Merge temporarily to ensure consistent encoding across train and test
df_full = pd.concat([df_train, df_test])

# Encode categorical columns as integers
cat_cols = ['protocol_type', 'service', 'flag']
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df_full[col] = le.fit_transform(df_full[col])
    label_encoders[col] = le

# map attacks to categories -------------------------------------------------------------------------------------------
# ⚠️ kept getting an error here, modified mappings from strings to int for XGBoost
category_map = {
    'normal': 0,
    # DoS
    'neptune': 1, 'back': 1, 'land': 1, 'pod': 1,
    'smurf': 1, 'teardrop': 1, 'mailbomb': 1, 'apache2': 1,
    'processtable': 1, 'udpstorm': 1, 'worm': 1,
    # Probe
    'satan': 2, 'ipsweep': 2, 'nmap': 2, 'portsweep': 2,
    'mscan': 2, 'saint': 2,
    # R2L
    'warezclient': 3, 'guess_passwd': 3, 'ftp_write': 3,
    'imap': 3, 'phf': 3, 'multihop': 3, 'warezmaster': 3,
    'spy': 3, 'xlock': 3, 'xsnoop': 3, 'snmpguess': 3,
    'snmpgetattack': 3, 'httptunnel': 3, 'sendmail': 3, 'named': 3,
    # U2R
    'buffer_overflow': 4, 'loadmodule': 4, 'rootkit': 4,
    'perl': 4, 'sqlattack': 4, 'xterm': 4, 'ps': 4
}

df_full['category'] = df_full['class'].map(category_map).fillna('Other')

# prepare features and labels -----------------------------------------------------------------------------------------
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

#print(f"\nFeatures: {x_train.shape[1]}")
#print(f"\nTraining set class distribution:")
#print(y_train.value_counts())
#print(f"\nTest set class distribution:")
#print(y_test.value_counts())

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

# calculate the scale_pos_weight
# taken directly from documentation, I don't think this is currently modeled correctly, needs more investigation
# is it y0 / (y1 + y2 + y3 + y4)?
scale_pos_weight = len(y_train[y_train == 0]) / (len(y_train[y_train == 1]) + len(y_train[y_train == 2]) + len(y_train[y_train == 3]) + len(y_train[y_train == 4]))

n_estimators = int(args.n_estimators)
max_depth = int(args.max_depth)
learning_rate = float(args.learning_rate)
subsample = float(args.subsample)

# --- SMOTE: generate synthetic examples for rare classes (R2L, U2R) ---
from imblearn.over_sampling import SMOTE

print("\n🧪 Applying SMOTE to balance training data...")
print("Before SMOTE:")
print(y_train.value_counts())

smote = SMOTE(random_state=42, k_neighbors=5)
x_train, y_train = smote.fit_resample(x_train, y_train)

print("\nAfter SMOTE:")
print(y_train.value_counts())

model = xgb.XGBClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    subsample=subsample,
    colsample_bytree=0.8,
    #scale_pos_weight=scale_pos_weight,
    objective='multi:softprob',          # for multiclass, dataset
    num_class = len(np.unique(y_train)), # specify number of multiclass for above
    eval_metric='merror',
    random_state=42,                     # keep static for reproducibility, per assignment
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

# evaluate
print(f"\n🎛️ Parameters: n_estimators={args.n_estimators}, max_depth={args.max_depth}, learning_rate={args.learning_rate}, subsample={args.subsample}")
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("📃 Classification Report:")
print("⮡  legend: 0=Normal, 1=DoS, 2=Probe, 3=R2L, 4=U2R\n")
print(classification_report(y_test, y_pred))
print(f"🎖️ F1:  \033[92m{f1_score(y_test, y_pred, average='macro')}\033[0m\n")

scores = cross_val_score(model, x_train, y_train, cv=5, scoring='f1_macro')
print(f"🔀 Val: \033[95m{scores.mean():.4f} (± {scores.std():.4f})\033[0m\n")

create_confusion_matrix(y_test, y_pred)

# save model
#model.save_model("intrusion_model.json")