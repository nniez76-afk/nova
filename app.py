import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Campus Facility Reservation Analysis", page_icon="🏫", layout="wide")

BASE_COLUMNS = [
    "Reservation_ID", "User_Category", "Facility_Type", "Lead_Time_Days",
    "Participants", "Actual_Attendance", "Usage_Status", "Department", "Time_Slot"
]
ENGINEERED_COLUMNS = ["Lead_Time_Category", "Reservation_Outcome", "Attendance_Ratio"]


@st.cache_data
def generate_sample_data(n=400):
    rng = np.random.default_rng(42)
    data = pd.DataFrame({
        "Reservation_ID": [f"RES-{1000+i}" for i in range(n)],
        "User_Category": rng.choice(["Student", "Faculty", "Staff", None], n, p=[.5, .25, .2, .05]),
        "Facility_Type": rng.choice(["Auditorium", "Discussion Room", "Computer Lab", "Lecture Hall"], n),
        "Lead_Time_Days": rng.integers(0, 30, n),
        "Participants": rng.integers(5, 100, n),
        "Actual_Attendance": rng.integers(0, 100, n),
        "Usage_Status": rng.choice(["Used", "Cancelled", "No-Show"], n, p=[.6, .25, .15]),
        "Department": rng.choice(["College of CS", "College of Eng", "College of Bus", "Admin"], n),
        "Time_Slot": rng.choice(["Morning", "Afternoon", "Evening"], n),
    })
    data["Actual_Attendance"] = np.minimum(data["Actual_Attendance"], data["Participants"])
    return data


def read_csv_safely(file):
    errors = []
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc), enc
        except UnicodeDecodeError as e:
            errors.append(str(e))
    raise ValueError("Cannot read this CSV. Please save it as CSV UTF-8 and upload again.")


def prepare_data(source):
    """Accept both the original 9-column CSV and the already-cleaned 12-column CSV."""
    df = source.copy()
    missing = [c for c in BASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    before = len(df)
    df = df.drop_duplicates(subset=["Reservation_ID"]).copy()
    duplicates = before - len(df)

    for c in ["Lead_Time_Days", "Participants", "Actual_Attendance"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["User_Category"] = df["User_Category"].astype("string").str.strip().str.capitalize()
    df["User_Category"] = df["User_Category"].fillna("Unknown").replace("", "Unknown")

    for c in ["Facility_Type", "Usage_Status", "Department", "Time_Slot"]:
        df[c] = df[c].astype("string").str.strip().fillna("Unknown").replace("", "Unknown")

    df["Lead_Time_Days"] = df["Lead_Time_Days"].fillna(0).clip(lower=0)
    df["Participants"] = df["Participants"].fillna(0).clip(lower=0)
    df["Actual_Attendance"] = df["Actual_Attendance"].fillna(0).clip(lower=0)
    df["Actual_Attendance"] = np.minimum(df["Actual_Attendance"], df["Participants"])

    # Rebuild derived fields from the base columns to keep them consistent.
    def lead_category(days):
        if days <= 2:
            return "Last Minute"
        if days <= 7:
            return "Short Lead"
        if days <= 14:
            return "Medium Lead"
        return "Early Booking"

    df["Lead_Time_Category"] = df["Lead_Time_Days"].apply(lead_category)
    df["Reservation_Outcome"] = df["Usage_Status"].str.lower().apply(
        lambda x: "Completed" if x == "used" else "Not Completed"
    )
    df["Attendance_Ratio"] = np.where(
        df["Participants"] > 0,
        df["Actual_Attendance"] / df["Participants"],
        0.0
    )
    return df, {"before": before, "after": len(df), "duplicates": duplicates}


def train_decision_tree(df):
    features = [
        "User_Category", "Facility_Type", "Lead_Time_Days",
        "Department", "Time_Slot", "Participants"
    ]
    X = df[features].copy()
    y = df["Reservation_Outcome"].map({"Completed": 0, "Not Completed": 1})

    if y.isna().any() or y.nunique() < 2 or y.value_counts().min() < 2:
        raise ValueError("The dataset must contain at least two outcome classes, with at least two rows per class.")

    categorical = ["User_Category", "Facility_Type", "Department", "Time_Slot"]
    numeric = ["Lead_Time_Days", "Participants"]
    prep = ColumnTransformer([
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical),
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric)
    ])
    model = Pipeline([
        ("preprocessor", prep),
        ("classifier", DecisionTreeClassifier(max_depth=4, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-Score": f1_score(y_test, pred, zero_division=0),
    }
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    report = classification_report(
        y_test, pred, labels=[0, 1],
        target_names=["Completed", "Not Completed"],
        zero_division=0, output_dict=True
    )
    names = model.named_steps["preprocessor"].get_feature_names_out()
    importance = pd.Series(
        model.named_steps["classifier"].feature_importances_, index=names
    ).sort_values(ascending=False)
    return metrics, cm, report, importance


st.title("🏫 Campus Facility Reservation Data Mining & Analysis")
st.write("Upload your reservation CSV or use the sample dataset to explore, clean, visualize, and model reservation outcomes.")

with st.sidebar:
    st.header("Data Source")
    source_option = st.radio("Choose your dataset", ["Use sample dataset", "Upload my CSV file"])
    uploaded = None
    if source_option == "Upload my CSV file":
        uploaded = st.file_uploader("Upload a reservation CSV", type=["csv"])

if source_option == "Use sample dataset":
    source_df = generate_sample_data()
    st.success("Sample dataset loaded.")
else:
    if uploaded is None:
        st.info("Upload a CSV file to begin, or select 'Use sample dataset'.")
        st.stop()
    try:
        source_df, encoding = read_csv_safely(uploaded)
        st.success(f"CSV loaded successfully. Detected encoding: {encoding}.")
    except Exception as e:
        st.error(f"CSV upload/read error: {e}")
        st.stop()

try:
    cleaned_df, summary = prepare_data(source_df)
except Exception as e:
    st.error(f"Dataset format problem: {e}")
    st.write("Required columns:", BASE_COLUMNS)
    st.stop()

overview, cleaning, analysis, model_tab, data_tab = st.tabs(
    ["Overview", "Data Cleaning", "Exploratory Analysis", "Decision Tree Model", "Dataset"]
)

with overview:
    st.subheader("Dataset Overview")
    a, b, c, d = st.columns(4)
    a.metric("Raw Records", f"{summary['before']:,}")
    b.metric("Cleaned Records", f"{summary['after']:,}")
    c.metric("Duplicates Removed", f"{summary['duplicates']:,}")
    d.metric("Columns", len(cleaned_df.columns))
    st.markdown("**Dataset preview**")
    st.dataframe(source_df.head(10), use_container_width=True)
    st.markdown("**Dataset summary**")
    st.dataframe(source_df.describe(include="all").transpose(), use_container_width=True)

with cleaning:
    st.subheader("Data Cleaning and Feature Engineering")
    a, b, c = st.columns(3)
    a.metric("Rows Before", summary["before"])
    b.metric("Rows After", summary["after"])
    c.metric("Duplicates Removed", summary["duplicates"])
    st.write("The app removes duplicate reservation IDs, handles missing values, fixes invalid numeric values, and creates the three derived columns.")
    st.dataframe(cleaned_df.head(20), use_container_width=True)
    st.download_button(
        "Download Cleaned Dataset (CSV)",
        cleaned_df.to_csv(index=False).encode("utf-8"),
        "facility_reservations_cleaned.csv",
        "text/csv"
    )

with analysis:
    st.subheader("Exploratory Data Analysis")
    counts = cleaned_df["Facility_Type"].value_counts()
    st.markdown("**1. Which facility type has the highest reservation count?**")
    if not counts.empty:
        st.write(f"Highest: **{counts.idxmax()}** ({counts.max()} reservations).")
        fig, ax = plt.subplots(figsize=(9, 4))
        counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Facility Type")
        ax.set_ylabel("Reservations")
        ax.set_title("Reservation Count by Facility Type")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("**2. What is the failure rate by user category?**")
    rates = cleaned_df.groupby("User_Category")["Reservation_Outcome"].apply(
        lambda s: (s == "Not Completed").mean() * 100
    ).reset_index(name="Failure_Rate_Percent").sort_values("Failure_Rate_Percent", ascending=False)
    st.dataframe(rates, use_container_width=True)

    st.markdown("**3. How does lead time relate to reservation outcome?**")
    cross = pd.crosstab(
        cleaned_df["Lead_Time_Category"], cleaned_df["Reservation_Outcome"], normalize="index"
    ) * 100
    st.dataframe(cross.round(2), use_container_width=True)
    if not cross.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        cross.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Reservation Outcome by Lead-Time Category")
        ax.set_xlabel("Lead-Time Category")
        ax.set_ylabel("Percentage")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with model_tab:
    st.subheader("Decision Tree Classification")
    st.write("The model does not use Usage_Status or Actual_Attendance as predictors, because these are known after the reservation outcome.")
    try:
        metrics, cm, report, importance = train_decision_tree(cleaned_df)
        cols = st.columns(4)
        for col, (label, value) in zip(cols, metrics.items()):
            col.metric(label, f"{value:.3f}")
        st.markdown("**Confusion Matrix**")
        st.dataframe(pd.DataFrame(
            cm,
            index=["Actual: Completed", "Actual: Not Completed"],
            columns=["Predicted: Completed", "Predicted: Not Completed"]
        ), use_container_width=True)
        st.markdown("**Classification Report**")
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
        st.markdown("**Top 3 Influential Predictors**")
        top = importance.head(3).rename("Importance").reset_index()
        top.columns = ["Predictor", "Importance"]
        st.dataframe(top, use_container_width=True)
        fig, ax = plt.subplots(figsize=(9, 4))
        importance.head(10).sort_values().plot(kind="barh", ax=ax)
        ax.set_title("Top 10 Feature Importances")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Model could not be trained: {e}")

with data_tab:
    st.subheader("View Dataset")
    choice = st.radio("Select dataset", ["Uploaded/Raw Dataset", "Cleaned Dataset"], horizontal=True)
    shown = source_df if choice == "Uploaded/Raw Dataset" else cleaned_df
    st.write(f"{len(shown):,} records")
    st.dataframe(shown, use_container_width=True, height=450)
    st.download_button(
        "Download Displayed Dataset",
        shown.to_csv(index=False).encode("utf-8"),
        "facility_reservations_raw.csv" if choice == "Uploaded/Raw Dataset" else "facility_reservations_cleaned.csv",
        "text/csv"
    )

st.divider()
st.caption("Campus Facility Reservation Data Mining & Analysis")
