import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from xgboost import XGBClassifier

import streamlit as st


SELECTED_FEATURES = [
    "studytime",
    "failures",
    "absences",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "higher",
    "internet",
    "romantic",
    "health",
    "goout",
    "Dalc",
    "Walc",
    "age"
]

TARGET = "pass"

def build_preprocessor(X):
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", "passthrough", categorical_features)
        ]
    )
    return preprocessor


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    return metrics


def save_model(model, filename):
    joblib.dump(model, filename)

def load_model(filename):
    return joblib.load(filename)


#Logitic Regression
def train_logistic_regression(X_train, y_train, preprocessor):
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)
    return model


#Decision Tree Classifier
def train_decision_tree(X_train, y_train, preprocessor):
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ])
    model.fit(X_train, y_train)
    return model

#KNN Classifier
def train_knn(X_train, y_train, preprocessor):
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ])
    model.fit(X_train, y_train)
    return model

#Gaussian Naive Bayes

def train_naive_bayes(X_train, y_train, preprocessor):
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GaussianNB())
    ])
    model.fit(X_train, y_train)
    return model

#Random Forest
def train_random_forest(X_train, y_train, preprocessor):
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ))
    ])
    model.fit(X_train, y_train)
    return model


#XGBoost

def train_xgboost(X_train, y_train, preprocessor):
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        ))
    ])
    model.fit(X_train, y_train)
    return model

def clean_student_dataset(df):
    """
    Cleans student-por dataset:
    - Converts yes/no to 1/0
    - Creates binary target from G3
    - Drops grade leakage columns
    """

    # ----------------------------------
    # YES/NO → 1/0 MAPPING
    # ----------------------------------
    yes_no_cols = [
        "schoolsup", "famsup", "paid", "activities",
        "higher", "internet", "romantic"
    ]

    for col in yes_no_cols:
        df[col] = df[col].map({"yes": 1, "no": 0})

    # ----------------------------------
    # TARGET CREATION
    # ----------------------------------
    df["pass"] = (df["G3"] >= 10).astype(int)

    # ----------------------------------
    # DROP LEAKAGE COLUMNS
    # ----------------------------------
    df = df.drop(columns=["G1", "G2", "G3"])

    return df




def test_and_save_all_models(X_test, y_test):
    results = []

    for model_file in os.listdir("saved_models"):
        if not model_file.endswith(".pkl"):
            continue

        model_name = model_file.replace(".pkl", "")
        print(f"Testing {model_name}...")

        model = joblib.load(f"saved_models/{model_file}")

        # REUSE EXISTING FUNCTION
        scores = evaluate_model(model, X_test, y_test)

        scores["Model"] = model_name
        results.append(scores)

    results_df = pd.DataFrame(results)
    results_df.to_csv("model_scores.csv", index=False)

    print("✅ Model evaluation complete. Scores saved to model_scores.csv")





def train_and_save_all_models(X_train, y_train, preprocessor):
    os.makedirs("saved_models", exist_ok=True)

    models = {
        "logistic_regression": train_logistic_regression,
        "decision_tree": train_decision_tree,
        "knn": train_knn,
        "naive_bayes": train_naive_bayes,
        "random_forest": train_random_forest,
        "xgboost": train_xgboost
    }

    for model_name, train_fn in models.items():
        print(f"Training {model_name}...")

        model = train_fn(X_train, y_train, preprocessor)

        save_path = f"saved_models/{model_name}.pkl"
        joblib.dump(model, save_path)

        print(f"Saved → {save_path}")

    print("✅ All models trained and saved.")

# if __name__=="__main__":
#     print("Reading Dataset File..")

#     df = pd.read_csv("student-por.csv")


#     # Create binary target
#     print(df.columns)
#     df["pass"] = (df["G3"] >= 10).astype(int)
#     df = clean_student_dataset(df)

#     print("Converting Dataset TARGET")


#     print(df["pass"].value_counts())


#     X = df[SELECTED_FEATURES]
#     y = df[TARGET]
#     print("SELECTED FEATURES",SELECTED_FEATURES)
#     print("SELECTED TARGET,",TARGET)
#     X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42
#     )

#     preprocessor = build_preprocessor(X)
#     train_and_save_all_models(X_train, y_train, preprocessor)
#     test_and_save_all_models(X_test, y_test)


st.set_page_config(
    page_title="Student Performance Prediction",
    layout="wide"
)

st.title("🎓 Student Performance Prediction App")
st.write("Upload test data and evaluate trained ML models.")


uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

missing_cols = set(SELECTED_FEATURES + [TARGET]) - set(df.columns)

if missing_cols:
    st.error(f"Missing columns: {missing_cols}")
    st.stop()

model_files = [
    f for f in os.listdir("saved_models") if f.endswith(".pkl")
]

if not model_files:
    st.error("No saved models found in 'saved_models/' directory.")
    st.stop()

model_name = st.selectbox(
    "Select a Model",
    model_files
)

model_path = os.path.join("saved_models", model_name)
model = joblib.load(model_path)

X_test = df[SELECTED_FEATURES]
y_test = df[TARGET]

metrics = evaluate_model(model, X_test, y_test)

st.subheader("📈 Evaluation Metrics")

col1, col2, col3 = st.columns(3)

for i, (k, v) in enumerate(metrics.items()):
    [col1, col2, col3][i % 3].metric(k, f"{v:.4f}")

from sklearn.metrics import confusion_matrix, classification_report

y_pred = model.predict(X_test)

st.subheader("🧮 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
st.write(cm)

st.subheader("📄 Classification Report")
st.text(classification_report(y_test, y_pred))
