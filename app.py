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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

st.set_page_config(
    page_title="Campus Facility Reservation Analysis",
    page_icon="🏫",
    layout="wide",
)

REQUIRED_COLUMNS = [
    "Reservation_ID",
    "User_Category",
    "Facility_Type",
    "Lead_Time_Days",
    "Participants",
    "Actual_Attendance",
    "Usage_Status",
    "Department",
    "Time_Slot",
]


@st.cache_data
def generate_sample_data(n_records=400):
    """Create the same type of sample data used in the original notebook."""
    np.random.seed(42)

    reservation_ids = [f"RES-{1000 + i}" for i in range(n_records)]
    if n_records > 50:
        reservation_ids[10] = reservation_ids[5]
        reservation_ids[50] = reservation_ids[45]

    user_categories = np.random.choice(
        ["Student", "Faculty", "Staff", "student", "FACULTY", None],
        size=n_records,
        p=[0.4, 0.25, 0.15, 0.1, 0.05, 0.05],
    )
    facility_types = np.random.choice(
        ["Auditorium", "Discussion Room", "Computer Lab", "Lecture Hall"],
        size=n_records,
    )
    lead_times = np.random.randint(-2, 30, size=n_records)
    participants = np.random.randint(5, 100, size=n_records)
    actual_attendance = participants - np.random.randint(0, 15, size=n_records)

    if n_records > 15:
        actual_attendance[15] = -5
    if n_records > 25:
        actual_attendance[25] = 120

    usage_statuses = np.random.choice(
        ["Used", "Cancelled", "No-Show"],
        size=n_records,
        p=[0.6, 0.25, 0.15],
    )
    departments = np.random.choice(
        ["College of CS", "College of Eng", "College of Bus", "Admin"],
        size=n_records,
    )
    time_slots = np.random.choice(
        ["Morning", "Afternoon", "Evening"],
        size=n_records,
    )

    return pd.DataFrame(
        {
            "Reservation_ID": reservation_ids,
            "User_Category": user_categories,
            "Facility_Type": facility_types,
            "Lead_Time_Days": lead_times,
            "Participants": participants,
            "Actual_Attendance": actual_attendance,
            "Usage_Status": usage_statuses,
            "Department": departments,
            "Time_Slot": time_slots,
        }
    )


def read_uploaded_csv(uploaded_file):
    """Try common encodings to avoid common CSV UnicodeDecodeError issues."""
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=encoding), encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("Could not decode this CSV file. Please save it as UTF-8 and upload again.")


def clean_data(raw_df):
    """Apply the cleaning and feature-engineering steps from the notebook."""
    df = raw_df.copy()
    original_rows = len(df)

    # Convert numeric fields safely.
    for col in ["Lead_Time_Days", "Participants", "Actual_Attendance"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove duplicate reservation IDs.
    df = df.drop_duplicates(subset=["Reservation_ID"]).copy()
    duplicates_removed = original_rows - len(df)

    # Normalize user category and fill missing categories.
    df["User_Category"] = (
        df["User_Category"]
        .astype("string")
        .str.strip()
        .str.capitalize()
        .fillna("Unknown")
    )
    df["User_Category"] = df["User_Category"].replace("", "Unknown")

    # Fill or repair numeric values.
    df["Lead_Time_Days"] = df["Lead_Time_Days"].fillna(0).clip(lower=0)
    df["Participants"] = df["Participants"].fillna(0).clip(lower=0)
    df["Actual_Attendance"] = df["Actual_Attendance"].fillna(0).clip(lower=0)
    df["Actual_Attendance"] = np.minimum(
        df["Actual_Attendance"], df["Participants"]
    )

    # Fill missing categorical fields so charts and model can use them.
    for col in ["Facility_Type", "Usage_Status", "Department", "Time_Slot"]:
        df[col] = df[col].astype("string").str.strip().fillna("Unknown")
        df[col] = df[col].replace("", "Unknown")

    def categorize_lead_time(days):
        if days <= 2:
            return "Last Minute"
        if days <= 7:
            return "Short Lead"
        if days <= 14:
            return "Medium Lead"
        return "Early Booking"

    df["Lead_Time_Category"] = df["Lead_Time_Days"].apply(categorize_lead_time)
    df["Reservation_Outcome"] = df["Usage_Status"].apply(
        lambda value: "Completed" if str(value).strip().lower() == "used"
        else "Not Completed"
    )
    df["Attendance_Ratio"] = np.where(
        df["Participants"] > 0,
        df["Actual_Attendance"] / df["Participants"],
        0.0,
    )

    summary = {
        "original_rows": original_rows,
        "cleaned_rows": len(df),
        "duplicates_removed": duplicates_removed,
    }
    return df, summary


def build_model(df):
    """Train a Decision Tree using only information available at booking time."""
    predictors = [
        "User_Category",
        "Facility_Type",
        "Lead_Time_Days",
        "Department",
        "Time_Slot",
        "Participants",
    ]
    target = df["Reservation_Outcome"].map(
        {"Completed": 0, "Not Completed": 1}
    )

    X = df[predictors].copy()
    categorical = [
        "User_Category",
        "Facility_Type",
        "Department",
        "Time_Slot",
    ]
    numeric = ["Lead_Time_Days", "Participants"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[("imputer", SimpleImputer(strategy="median"))]
                ),
                numeric,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(max_depth=4, random_state=42)),
        ]
    )

    if target.nunique() < 2 or target.value_counts().min() < 2:
        raise ValueError(
            "The dataset needs at least two reservation outcome classes, "
            "with at least two records in each class, to train the model."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-Score": f1_score(y_test, predictions, zero_division=0),
    }
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Completed", "Not Completed"],
        zero_division=0,
        output_dict=True,
    )

    # Get feature names after preprocessing.
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = pd.Series(
        model.named_steps["classifier"].feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False)

    return metrics, matrix, report, importances, model


st.title("🏫 Campus Facility Reservation Data Mining & Analysis")
st.write(
    "Explore campus facility reservations, clean the data, view patterns, "
    "and evaluate a Decision Tree model that predicts reservation non-completion risk."
)

with st.sidebar:
    st.header("Data Source")
    source = st.radio(
        "Choose your dataset",
        ["Use sample dataset", "Upload my CSV file"],
    )
    uploaded_file = None
    if source == "Upload my CSV file":
        uploaded_file = st.file_uploader(
            "Upload a reservation CSV",
            type=["csv"],
            help="Your CSV must contain the required reservation columns.",
        )

    st.divider()
    st.caption("The sample dataset contains 400 generated reservation records.")

if source == "Use sample dataset":
    raw_df = generate_sample_data()
    st.success("Sample dataset loaded.")
elif uploaded_file is not None:
    try:
        raw_df, detected_encoding = read_uploaded_csv(uploaded_file)
        st.success(f"CSV uploaded successfully (encoding: {detected_encoding}).")
    except Exception as exc:
        st.error(f"Unable to read the CSV: {exc}")
        st.stop()
else:
    st.info("Upload a CSV file to begin, or select 'Use sample dataset'.")
    st.stop()

missing_columns = [col for col in REQUIRED_COLUMNS if col not in raw_df.columns]
if missing_columns:
    st.error(
        "Your CSV is missing these required columns: "
        + ", ".join(missing_columns)
    )
    st.write("Required columns:", REQUIRED_COLUMNS)
    st.stop()

raw_df = raw_df[REQUIRED_COLUMNS].copy()
clean_df, cleaning_summary = clean_data(raw_df)

tab_overview, tab_cleaning, tab_eda, tab_model, tab_data = st.tabs(
    ["Overview", "Data Cleaning", "Exploratory Analysis", "Decision Tree Model", "Dataset"]
)

with tab_overview:
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Raw Records", f"{len(raw_df):,}")
    col2.metric("Cleaned Records", f"{len(clean_df):,}")
    col3.metric("Duplicates Removed", f"{cleaning_summary['duplicates_removed']:,}")
    col4.metric("Number of Columns", f"{len(clean_df.columns)}")

    st.markdown("**Raw dataset preview**")
    st.dataframe(raw_df.head(10), use_container_width=True)

    st.markdown("**Raw dataset summary**")
    st.dataframe(raw_df.describe(include="all").transpose(), use_container_width=True)

with tab_cleaning:
    st.subheader("Data Cleaning and Feature Engineering")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows Before Cleaning", cleaning_summary["original_rows"])
    c2.metric("Rows After Cleaning", cleaning_summary["cleaned_rows"])
    c3.metric("Duplicates Removed", cleaning_summary["duplicates_removed"])

    st.markdown(
        """
        The application applies these steps:
        - Removes duplicate `Reservation_ID` values.
        - Standardizes `User_Category` and replaces missing categories with `Unknown`.
        - Changes negative or missing lead times to zero.
        - Replaces invalid attendance values and limits attendance to the participant count.
        - Creates `Lead_Time_Category`, `Reservation_Outcome`, and `Attendance_Ratio`.
        """
    )
    st.markdown("**Cleaned dataset preview**")
    st.dataframe(clean_df.head(20), use_container_width=True)

    st.download_button(
        "⬇️ Download Cleaned Dataset (CSV)",
        data=clean_df.to_csv(index=False).encode("utf-8"),
        file_name="facility_reservations_cleaned.csv",
        mime="text/csv",
    )

with tab_eda:
    st.subheader("Exploratory Data Analysis")
    st.markdown("**Question 1: Which facility type has the highest reservation count?**")
    facility_counts = clean_df["Facility_Type"].value_counts()
    if not facility_counts.empty:
        st.write(
            f"Highest booked facility: **{facility_counts.idxmax()}** "
            f"({facility_counts.max()} reservations)."
        )
        fig1, ax1 = plt.subplots(figsize=(9, 4.5))
        facility_counts.plot(kind="bar", ax=ax1)
        ax1.set_title("Reservation Count by Facility Type")
        ax1.set_xlabel("Facility Type")
        ax1.set_ylabel("Total Reservations")
        ax1.tick_params(axis="x", rotation=25)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    st.markdown("**Question 2: What is the failure rate by user category?**")
    user_failure = (
        clean_df.groupby("User_Category")["Reservation_Outcome"]
        .apply(lambda values: (values == "Not Completed").mean() * 100)
        .reset_index(name="Failure_Rate_Pct")
        .sort_values("Failure_Rate_Pct", ascending=False)
    )
    st.dataframe(
        user_failure.style.format({"Failure_Rate_Pct": "{:.2f}%"}),
        use_container_width=True,
    )

    st.markdown("**Question 3: How does lead time relate to reservation outcome?**")
    lead_outcome = pd.crosstab(
        clean_df["Lead_Time_Category"],
        clean_df["Reservation_Outcome"],
        normalize="index",
    ) * 100
    st.dataframe(lead_outcome.round(2), use_container_width=True)

    if not lead_outcome.empty:
        fig2, ax2 = plt.subplots(figsize=(9, 4.5))
        lead_outcome.plot(kind="bar", stacked=True, ax=ax2)
        ax2.set_title("Reservation Outcome by Lead-Time Category")
        ax2.set_xlabel("Lead-Time Category")
        ax2.set_ylabel("Percentage (%)")
        ax2.tick_params(axis="x", rotation=0)
        ax2.legend(title="Outcome")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.caption(
        "Failure rate combines Cancelled and No-Show records as Not Completed, "
        "matching the outcome definition in the notebook."
    )

with tab_model:
    st.subheader("Decision Tree Classification")
    st.write(
        "The model uses only information available when a reservation is submitted. "
        "Usage_Status and Actual_Attendance are excluded from the predictors to reduce data leakage."
    )
    try:
        metrics, matrix, report, importances, trained_model = build_model(clean_df)

        metric_cols = st.columns(4)
        for column, (name, value) in zip(metric_cols, metrics.items()):
            column.metric(name, f"{value:.3f}")

        st.markdown("**Confusion Matrix**")
        st.write("Rows = actual class; columns = predicted class.")
        st.dataframe(
            pd.DataFrame(
                matrix,
                index=["Actual: Completed", "Actual: Not Completed"],
                columns=["Predicted: Completed", "Predicted: Not Completed"],
            ),
            use_container_width=True,
        )

        st.markdown("**Classification Report**")
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

        st.markdown("**Top 3 Most Influential Predictors**")
        top3 = importances.head(3).rename("Importance").reset_index()
        top3.columns = ["Predictor", "Importance"]
        st.dataframe(top3, use_container_width=True)

        fig3, ax3 = plt.subplots(figsize=(9, 4.5))
        importances.head(10).sort_values().plot(kind="barh", ax=ax3)
        ax3.set_title("Top 10 Feature Importances")
        ax3.set_xlabel("Importance")
        ax3.set_ylabel("Predictor")
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        st.caption(
            "Model results depend on the uploaded or generated dataset. "
            "They are evaluation results, not a guarantee of future reservation outcomes."
        )
    except Exception as exc:
        st.warning(f"The model could not be trained with this dataset: {exc}")

with tab_data:
    st.subheader("View Dataset")
    view_choice = st.radio(
        "Select dataset to display",
        ["Raw Dataset", "Cleaned Dataset"],
        horizontal=True,
    )
    selected_df = raw_df if view_choice == "Raw Dataset" else clean_df
    st.write(f"Showing {len(selected_df):,} records.")
    st.dataframe(selected_df, use_container_width=True, height=450)

    st.download_button(
        "⬇️ Download Displayed Dataset",
        data=selected_df.to_csv(index=False).encode("utf-8"),
        file_name=(
            "facility_reservations_raw.csv"
            if view_choice == "Raw Dataset"
            else "facility_reservations_cleaned.csv"
        ),
        mime="text/csv",
    )

st.divider()
st.caption("Campus Facility Reservation Data Mining & Analysis | Streamlit Application")
