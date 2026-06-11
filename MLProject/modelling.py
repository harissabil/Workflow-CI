import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

TARGET_COL = "target"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Heart Disease RF Training for MLflow CI")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--data_path", type=str, default="heart_disease_preprocessing.csv")
    parser.add_argument("--test_size", type=float, default=TEST_SIZE)
    parser.add_argument("--random_state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def load_and_split(data_path: str, test_size: float, random_state: int):
    df = pd.read_csv(data_path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def main():
    args = parse_args()

    mlflow.set_experiment("heart_disease_ci")
    mlflow.sklearn.autolog()

    X_train, X_test, y_train, y_test = load_and_split(
        args.data_path, args.test_size, args.random_state
    )

    # mlflow run sets MLFLOW_RUN_ID before calling this script;
    # autolog will attach to that active run automatically — no start_run() needed.
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Heart Disease"]))


if __name__ == "__main__":
    main()
