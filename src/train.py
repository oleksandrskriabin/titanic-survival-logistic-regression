import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# -----------------------------
# Feature engineering helpers
# -----------------------------
TITLE_MAP = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
    "Dr": "Officer",
    "Rev": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Capt": "Officer",
    "Sir": "Royalty",
    "Lady": "Royalty",
    "Countess": "Royalty",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Dona": "Royalty",
}


def add_cabin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create Cabin_missing flag and Deck feature, then drop Cabin."""
    df = df.copy()
    df["Cabin_missing"] = df["Cabin"].isna().astype(int)
    df["Deck"] = df["Cabin"].str[0].fillna("Unknown")
    df = df.drop(columns=["Cabin"])
    return df


def add_title_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Extract Title from Name, normalize rare titles, then drop Name."""
    df = df.copy()
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")
    df["Title"] = df["Title"].replace(TITLE_MAP)
    df = df.drop(columns=["Name"])
    return df


def fill_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing Embarked with the most common port (S)."""
    df = df.copy()
    df["Embarked"] = df["Embarked"].fillna("S")
    return df


def fit_age_imputer(train_df: pd.DataFrame):
    """
    Learn median Age grouped by (Sex, Pclass) from TRAIN ONLY.
    Returns a function that imputes Age for any dataframe row.
    """
    group_medians = train_df.groupby(["Sex", "Pclass"])["Age"].median()
    global_median = train_df["Age"].median()

    def impute_age_row(row):
        if pd.isna(row["Age"]):
            key = (row["Sex"], row["Pclass"])
            # fallback to global median if group not found
            return group_medians.get(key, global_median)
        return row["Age"]

    return impute_age_row


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Apply identical feature engineering to train and test.
    Important: Age imputer is fit on train only, then applied to both.
    """
    train = add_cabin_features(train_df)
    test = add_cabin_features(test_df)

    train = fill_embarked(train)
    test = fill_embarked(test)

    train = add_title_feature(train)
    test = add_title_feature(test)

    impute_age_row = fit_age_imputer(train)
    train["Age"] = train.apply(impute_age_row, axis=1)
    test["Age"] = test.apply(impute_age_row, axis=1)

    return train, test


# -----------------------------
# Modeling
# -----------------------------
def build_model(numeric_features, categorical_features) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = LogisticRegression(max_iter=1000)

    clf = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model)
    ])
    return clf


def main():
    # Load data
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_DIR = BASE_DIR / "data"

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    # Feature engineering
    train_df, test_df = prepare_features(train_df, test_df)

    # Split features/target
    y = train_df["Survived"]
    X = train_df.drop(columns=["Survived"])

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Column groups
    numeric_features = ["Age", "Fare", "SibSp", "Parch", "Cabin_missing"]
    categorical_features = ["Sex", "Embarked", "Deck", "Pclass", "Title"]

    # Build + train
    clf = build_model(numeric_features, categorical_features)
    clf.fit(X_train, y_train)

    # Validate
    val_pred = clf.predict(X_val)
    print("Accuracy:", accuracy_score(y_val, val_pred))
    print("F1:", f1_score(y_val, val_pred))
    print("Confusion matrix:\n", confusion_matrix(y_val, val_pred))
    print("\nReport:\n", classification_report(y_val, val_pred))

    # Refit on full training data + predict test
    clf.fit(X, y)
    test_preds = clf.predict(test_df)

    # Create submission
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": test_preds
    })

    submission.to_csv("/Kaggle/Titanic/submission.csv", index=False)
    print("Saved submission to /Kaggle/Titanic/submission.csv")


if __name__ == "__main__":
    main()
