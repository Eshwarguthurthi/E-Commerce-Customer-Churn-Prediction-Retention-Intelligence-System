import os
import sys
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    FunctionTransformer
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import joblib


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ecommerce_churn_cleaned.csv"
)

ARTIFACT_DIR = os.path.join(
    BASE_DIR,
    "artifacts"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ============================================================
# IMPORT FEATURE ENGINEERING
# ============================================================

sys.path.append(
    os.path.join(BASE_DIR, "src")
)

from feature_engineering import create_features


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# TARGET / FEATURES
# ============================================================

y = df["Churn"]

X = df.drop(
    columns=["Churn", "CustomerID"]
)

print("Feature shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# FEATURE ENGINEERING
# ============================================================

X_train_features = create_features(X_train)

numerical_cols = X_train_features.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_cols = X_train_features.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumerical columns:")
print(numerical_cols)

print("\nCategorical columns:")
print(categorical_cols)


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numerical_cols
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_cols
        )
    ]
)


# ============================================================
# MODEL DEFINITIONS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            random_state=42
        )
}


# ============================================================
# MODEL EVALUATION
# ============================================================

results = []


for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    pipeline = Pipeline(
        steps=[
            (
                "feature_engineering",
                FunctionTransformer(
                    create_features,
                    validate=False
                )
            ),
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    y_pred = pipeline.predict(
        X_test
    )

    y_proba = pipeline.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_proba
    )

    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc
        }
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )
    print(
        f"Precision: {precision:.4f}"
    )
    print(
        f"Recall   : {recall:.4f}"
    )
    print(
        f"F1 Score : {f1:.4f}"
    )
    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )


# ============================================================
# SAVE MODEL COMPARISON
# ============================================================

comparison_df = pd.DataFrame(results)

comparison_path = os.path.join(
    PROCESSED_DIR,
    "model_comparison.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)

print(
    f"\nModel comparison saved to:\n{comparison_path}"
)


# ============================================================
# RANDOM FOREST TUNING
# ============================================================

print(
    "\nStarting Random Forest hyperparameter tuning..."
)


rf_pipeline = Pipeline(
    steps=[
        (
            "feature_engineering",
            FunctionTransformer(
                create_features,
                validate=False
            )
        ),
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


param_distributions = {

    "model__n_estimators": [
        100,
        200,
        300,
        500
    ],

    "model__max_depth": [
        None,
        5,
        10,
        15,
        20,
        30
    ],

    "model__min_samples_split": [
        2,
        5,
        10
    ],

    "model__min_samples_leaf": [
        1,
        2,
        4
    ],

    "model__max_features": [
        "sqrt",
        "log2",
        None
    ]
}


random_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=param_distributions,
    n_iter=20,
    cv=5,
    scoring="f1",
    random_state=42,
    n_jobs=-1,
    verbose=1
)


random_search.fit(
    X_train,
    y_train
)


print(
    "\nBest Parameters:"
)

print(
    random_search.best_params_
)


print(
    "\nBest Cross Validation F1:"
)

print(
    random_search.best_score_
)


# ============================================================
# FINAL PIPELINE
# ============================================================

final_pipeline = random_search.best_estimator_


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

y_pred = final_pipeline.predict(
    X_test
)

y_proba = final_pipeline.predict_proba(
    X_test
)[:, 1]


final_accuracy = accuracy_score(
    y_test,
    y_pred
)

final_precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

final_recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

final_f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

final_roc_auc = roc_auc_score(
    y_test,
    y_proba
)


print("\nFinal Model Performance")
print("=" * 40)

print(
    f"Accuracy : {final_accuracy:.4f}"
)

print(
    f"Precision: {final_precision:.4f}"
)

print(
    f"Recall   : {final_recall:.4f}"
)

print(
    f"F1 Score : {final_f1:.4f}"
)

print(
    f"ROC-AUC  : {final_roc_auc:.4f}"
)


# ============================================================
# SAVE FINAL PIPELINE
# ============================================================

model_path = os.path.join(
    ARTIFACT_DIR,
    "churn_pipeline.joblib"
)

joblib.dump(
    final_pipeline,
    model_path
)

print(
    f"\nFinal pipeline saved to:\n{model_path}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

rf_model = final_pipeline.named_steps[
    "model"
]

final_preprocessor = (
    final_pipeline
    .named_steps["preprocessor"]
)


feature_names = (
    final_preprocessor
    .get_feature_names_out()
)


importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": rf_model.feature_importances_
    }
)


importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


importance_path = os.path.join(
    PROCESSED_DIR,
    "feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)


print(
    f"\nFeature importance saved to:\n{importance_path}"
)

print(
    "\nTop 10 important features:"
)

print(
    importance_df.head(10)
)

print("\nTraining completed successfully.")