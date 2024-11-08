#newensemble

import sklearn
import xgboost as xgb
import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from scipy import stats
from scipy.stats import boxcox
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv(r"E:\PCOS_DATASET.csv")
df_pc = df.replace({',': '', '\.': '', "'": ''}, regex=True)
df_pc = df.apply(pd.to_numeric, errors='coerce')

# Drop columns with high percentage of missing values
columns_to_drop = ["Sl. No", "Patient File No.", "Marraige Status (Yrs)", "Blood Group", "RR (breaths/min)",
                   "Hb(g/dl)", "Cycle(R/I)", "Pregnant(Y/N)", "Hip(inch)", "PRL(ng/mL)",
                   "PRG(ng/mL)", "RBS(mg/dl)", "hair growth(Y/N)", "Skin darkening (Y/N)", 
                   "Hair loss(Y/N)", "Pimples(Y/N)", "Fast food (Y/N)", "Reg.Exercise(Y/N)", 
                   "BP _Systolic (mmHg)", "BP _Diastolic (mmHg)"]

df_new = df_pc.drop(columns=columns_to_drop, axis=1)

df_new.columns
# Identify categorical and numerical columns
categorical_columns = df_new.select_dtypes(include=['object', 'category']).columns
numerical_columns = df_new.select_dtypes(include=['number']).columns

# Impute missing values
imputer = SimpleImputer(strategy='median')
imputed_data = imputer.fit_transform(df_new)
df_cleaned = pd.DataFrame(imputed_data, columns=numerical_columns)

# Remove outliers
z_scores = np.abs(stats.zscore(df_cleaned, nan_policy='omit'))
outliers_threshold = 5
mask = (z_scores <= outliers_threshold).all(axis=1)
df_cleaned = df_cleaned[mask]

# Apply Box-Cox transformation
df_transformed = df_cleaned.copy()
for col in df_transformed.columns:
    if (df_transformed[col] > 0).all():  # Box-Cox requires positive values
        df_transformed[col], _ = boxcox(df_transformed[col] + 1)  # +1 to handle zero values

# Drop rows with missing values in specific columns
columns_to_drop_na = ['AMH(ng/mL)', '2beta-HCG(mIU/mL)']
df_transformed.dropna(subset=columns_to_drop_na, inplace=True)

# Define features and target
x = df_transformed.drop(columns=['PCOS (Y/N)'])
y = df_transformed['PCOS (Y/N)']

# Feature scaling
scaler = RobustScaler()
x_scaled = scaler.fit_transform(x)
x_scaled.shape###############

# Train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

x_train.shape#############

# Define ClassifierModel class
class ClassifierModel(object):
    def __init__(self, clf, params=None):
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)
    
    def fit(self, x, y):
        return self.clf.fit(x, y)
    
    def feature_importances(self, x, y):
        return self.clf.fit(x, y).feature_importances_
    
    def predict(self, x):
        return self.clf.predict(x)

# Define trainModel function
def trainModel(model, x_train, y_train, x_test, n_folds, seed):
    cv = KFold(n_splits=n_folds, random_state=seed, shuffle=True)
    scores = cross_val_score(model.clf, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
    y_pred = cross_val_predict(model.clf, x_train, y_train, cv=cv, n_jobs=-1)
    return scores, y_pred

# Train models and compute feature importances
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier

# Extra Trees Classifier
et_params = {
    'n_jobs': -1,
    'n_estimators': 400,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'random_state': 42
}
etc_model = ClassifierModel(clf=ExtraTreesClassifier, params=et_params)
etc_scores, etc_train_pred = trainModel(etc_model, x_train, y_train, x_test, 5, 45)
etc_features = etc_model.feature_importances(x_train, y_train)
etc_scores

# CatBoost Classifier
cat_params = {
    'iterations': 200,
    'learning_rate': 0.1,
    'depth': 6,
    'random_state': 42
}
cat_model = ClassifierModel(clf=CatBoostClassifier, params=cat_params)
cat_scores, cat_train_pred = trainModel(cat_model, x_train, y_train, x_test, 5, 42)
cat_features = cat_model.feature_importances(x_train, y_train)
cat_features

# AdaBoost Classifier
ada_params = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'random_state': 42
}
ada_model = ClassifierModel(clf=AdaBoostClassifier, params=ada_params)
ada_scores, ada_train_pred = trainModel(ada_model, x_train, y_train, x_test, 5, 42)
ada_features = ada_model.feature_importances(x_train, y_train)
ada_scores
# XGBoost Classifier

xgb_params = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'random_state': 42
}
xgb_model = ClassifierModel(clf=XGBClassifier, params=xgb_params)
xgb_scores, xgb_train_pred = trainModel(xgb_model, x_train, y_train, x_test, 5, 42)
xgb_features = xgb_model.feature_importances(x_train, y_train)
xgb_scores

# LightGBM Classifier
lgb_params = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'num_leaves': 31,
    'subsample': 0.8,
    'random_state': 42
}
lgb_model = ClassifierModel(clf=LGBMClassifier, params=lgb_params)
lgb_scores, lgb_train_pred = trainModel(lgb_model, x_train, y_train, x_test, 5, 42)
lgb_features = lgb_model.feature_importances(x_train, y_train)
lgb_scores
lgb_features

# Align feature importances
x.columns
feature_names = x.columns.values
def align_features(features_list, length):
    return [f[:length] if f.size > 0 else np.zeros(length) for f in features_list]

features_list = [etc_features, ada_features, cat_features, xgb_features, lgb_features]
aligned_features = align_features(features_list, x.shape[1])

# Create feature importance DataFrame
feature_dataframe = pd.DataFrame({
    'features': feature_names,
    'Extra Trees feature importances': aligned_features[0],
    'AdaBoost feature importances': aligned_features[1],
    'CatBoost feature importances': aligned_features[2],
    'XGBoost feature importances': aligned_features[3],
    'LightGBM feature importances': aligned_features[4]
})
feature_dataframe['mean'] = feature_dataframe[['Extra Trees feature importances', 
                                                'AdaBoost feature importances', 
                                                'CatBoost feature importances', 
                                                'XGBoost feature importances', 
                                                'LightGBM feature importances']].mean(axis=1)

print(feature_dataframe)

# Calculate model accuracies
def trainModel(model, x_train, y_train, x_test, n_folds, seed):
    cv = KFold(n_splits=n_folds, random_state=seed, shuffle=True)
    scores = cross_val_score(model.clf, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
    y_pred = cross_val_predict(model.clf, x_train, y_train, cv=cv, n_jobs=-1)
    return scores, y_pred

# Model accuracy DataFrame
acc_pred_train = pd.DataFrame({
    'ExtraTrees': etc_scores,
    'AdaBoost': ada_scores,
    'GradientBoost': xgb_scores,
    'catboost': cat_scores,
    'light gradientboost': lgb_scores
})

# Stack models
def trainStackModel(x_train, y_train, x_test, n_folds, seed):
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    gbm = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=0.8,
        random_state=seed
    ).fit(x_train, y_train)
    scores = cross_val_score(gbm, x_train, y_train, scoring='accuracy', cv=cv)
    return scores

stackModel_scores = trainStackModel(x_train, y_train, x_test, 5, 42)
acc_pred_train['stackingModel'] = stackModel_scores
acc_pred_train

x_train.shape
# Stacking Classifier
estimators = [
    ('catboost', cat_model.clf),
    ('adaboost', ada_model.clf),
    ('xgboost', xgb_model.clf),
    ('lightgbm', lgb_model.clf)
]

stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)

# Train the stacking ensemble
stacking_model.fit(x_train, y_train)

# Evaluate the stacking ensemble
stacking_scores = cross_val_score(stacking_model, x_train, y_train, cv=5, scoring='accuracy')

print("Stacking Model Scores: ", stacking_scores)
print("Stacking Model Mean Accuracy: ", stacking_scores.mean())



from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt

# Generate predictions on the test set
y_pred = stacking_model.predict(x_test)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plotting the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Compute the predicted probabilities for the positive class (PCOS)
y_pred_proba = stacking_model.predict_proba(x_test)[:, 1]

# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for reference
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# Prepare input features for prediction
features = [
    4.840525, 6.709318, 0.932051, 10.960228, 0.067106, 5.000000, 0.000000,
    1.509603, 0.840365, 1.654392, 0.533267, 1.073277, 12.411234, 13.693408,
    0.884021, 1.010000, 42.360000, 0.000000, 5.000000, 2.000000, 11.500000,
    4.700000, 6.655190
]

# Convert list to NumPy array and reshape
features_array = np.array(features).reshape(1, -1)

# Scale features using the same scaler
x_scaled_features = scaler.transform(features_array)

# Predict using the stacking model
prediction = stacking_model.predict(x_scaled_features)
print(f"Prediction: {prediction}")




import pickle

# Save the stacking model to a file
model_filename = 'stacking_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(stacking_model, file)
print(f"Model and scaler saved to {model_filename}")


scaler_filename = 'myscaler.pkl'
with open(scaler_filename, 'wb') as file:
    pickle.dump(scaler, file)
print(f"Scaler saved to {scaler_filename}")

import pickle

# Load the saved scaler
with open(scaler_filename, 'rb') as file:
    loaded_scaler = pickle.load(file)
print(f"Loaded scaler type: {type(loaded_scaler)}")
print(f"Scaler scale: {loaded_scaler.scale_}")
