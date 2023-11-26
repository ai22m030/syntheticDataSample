import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import ks_2samp

# Initialize label encoder
label_encoder = LabelEncoder()

# Load data
data = pd.read_csv('adult-subset-for-synthetic.csv')

# Encode the target variable
data['salary-class'] = label_encoder.fit_transform(data['salary-class'])

# Split data into features and target
X = data.drop('salary-class', axis=1)
y = data['salary-class']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2210585030)

# Automatically infer metadata from the DataFrame
metadata = SingleTableMetadata()

# Combine X_train and y_train for Gaussian Copula
train_data = pd.concat([X_train, y_train], axis=1)
metadata.detect_from_dataframe(data=train_data)

# Create and train the Gaussian Copula model
model = GaussianCopulaSynthesizer(metadata)
model.fit(train_data)

synthetic_data = model.sample(len(train_data))

# Compare distributions for a few selected columns
for column in train_data.columns:
    plt.figure(figsize=(10, 4))

    # Define the order for consistent plotting
    order = train_data[column].value_counts().index

    if train_data[column].dtype == 'object' or column == 'salary-class':
        # For categorical data, use count plot
        sns.countplot(x=column, data=train_data, order=order, color='blue', alpha=0.5, label='Original')
        sns.countplot(x=column, data=synthetic_data, order=order, color='red', alpha=0.5, label='Synthetic')
        plt.title(f'Distribution of {column}')

        # If the column is salary-class, set custom x-tick labels
        if column == 'salary-class':
            # Get the class labels using inverse_transform
            class_labels = label_encoder.inverse_transform(train_data['salary-class'].unique())
            plt.xticks(ticks=range(len(class_labels)), labels=class_labels, rotation=45)

        plt.legend()
    else:
        # For continuous data, use KDE plot
        sns.kdeplot(train_data[column], label='Original', fill=True, common_norm=False)
        sns.kdeplot(synthetic_data[column], label='Synthetic', fill=True, common_norm=False)
        plt.title(f'Distribution of {column}')
        plt.legend()

    plt.show()

# Calculate correlation matrices
original_corr = train_data.corr(numeric_only=True)
synthetic_corr = synthetic_data.corr(numeric_only=True)

# Plot heatmaps of the correlations
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.heatmap(original_corr, annot=True)
plt.title('Original Data Correlations')

plt.subplot(1, 2, 2)
sns.heatmap(synthetic_corr, annot=True)
plt.title('Synthetic Data Correlations')

plt.show()

# Kolmogorov-Smirnov test for each column
for column in train_data.columns:
    ks_stat, ks_p_value = ks_2samp(train_data[column], synthetic_data[column])
    print(f"{column} - KS Statistic: {ks_stat}, P-Value: {ks_p_value}")

# Define categorical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

# Create a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Preprocess original and synthetic data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
synthetic_data_processed = preprocessor.transform(synthetic_data.drop('salary-class', axis=1))
synthetic_target = synthetic_data['salary-class']

# Initialize models
lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier()

# Dictionary to store performance metrics
performance_metrics = {
    'Original Data': {},
    'Synthetic Data': {}
}

# Evaluate models and store metrics
for model in [lr, dt]:
    # Train on original data and store metrics
    model.fit(X_train_processed, y_train)
    original_predictions = model.predict(X_test_processed)
    performance_metrics['Original Data'][model.__class__.__name__] = {
        'Accuracy': accuracy_score(y_test, original_predictions),
        'F1': f1_score(y_test, original_predictions, average='macro'),
        'ROC AUC': roc_auc_score(y_test, original_predictions),
        'Precision': precision_score(y_test, original_predictions, average='binary'),
        'Recall': recall_score(y_test, original_predictions, average='binary')
    }

    # Train on synthetic data and store metrics
    model.fit(synthetic_data_processed, synthetic_target)
    synthetic_predictions = model.predict(X_test_processed)
    performance_metrics['Synthetic Data'][model.__class__.__name__] = {
        'Accuracy': accuracy_score(y_test, synthetic_predictions),
        'F1': f1_score(y_test, synthetic_predictions, average='macro'),
        'ROC AUC': roc_auc_score(y_test, synthetic_predictions),
        'Precision': precision_score(y_test, synthetic_predictions, average='binary'),
        'Recall': recall_score(y_test, synthetic_predictions, average='binary')
    }

# Reporting utility loss
for metric in ['Accuracy', 'F1', 'ROC AUC', 'Precision', 'Recall']:
    print(f"Utility Loss - {metric}:")
    for model_name in performance_metrics['Original Data']:
        original_metric = performance_metrics['Original Data'][model_name][metric]
        synthetic_metric = performance_metrics['Synthetic Data'][model_name][metric]
        utility_loss = original_metric - synthetic_metric
        print(f"  {model_name}: {utility_loss:.4f}")
    print()
