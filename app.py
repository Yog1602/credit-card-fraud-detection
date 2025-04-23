import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

DATA_PATH = r"C:\Users\Yog Dalal\Desktop\SE PRACTICAL\small_dataset.csv"
MODEL_PATH = "rf_fraud_model.joblib"

# Load & preprocess data
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def build_preprocessor(df: pd.DataFrame):
    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if "is_fraud" in numeric_features:
        numeric_features.remove("is_fraud")
    if "is_fraud" in categorical_features:
        categorical_features.remove("is_fraud")

    preprocessor = ColumnTransformer([
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])
    return preprocessor

# Train model
def train_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    preprocessor = build_preprocessor(df)
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=random_state,
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", clf),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    joblib.dump({"model": pipe, "lookup_cols": ["cc_num", "trans_num"]}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# Load model and make predictions
def load_model(path: str = MODEL_PATH):
    return joblib.load(path)

def predict_by_keys(cc_num: int, trans_num: str, df: pd.DataFrame, model_bundle):
    subset = df[(df["cc_num"] == cc_num) & (df["trans_num"] == trans_num)]
    if subset.empty:
        raise ValueError("No matching record found for the provided cc_num and trans_num.")

    X_query = subset.drop(columns=["is_fraud"])
    model = model_bundle["model"]
    pred = model.predict(X_query)[0]
    proba = model.predict_proba(X_query)[0][1]
    return int(pred), proba

# Interactive CLI mode
def main():
    df = load_data()
    print("Welcome to the Credit Card Fraud Detection System")
    choice = input("Do you want to train the model? (yes/no): ").strip().lower()

    if choice == "yes":
        train_model(df)
    else:
        model_bundle = load_model()

        while True:
            try:
                cc_num = int(input("Enter Credit Card Number: ").strip())
                trans_num = input("Enter Transaction Number: ").strip()
                pred, proba = predict_by_keys(cc_num, trans_num, df, model_bundle)
                label = "FRAUD" if pred == 1 else "LEGIT"
                print(f"Prediction: {label} (probability = {proba:.4f})")
            except ValueError as e:
                print(f"Error: {e}")

            again = input("Do you want to test another transaction? (yes/no): ").strip().lower()
            if again != "yes":
                break

if __name__ == "__main__":
    main()
