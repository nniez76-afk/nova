import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)

st.set_page_config(
    page_title="Student Service Request Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Student Service Request & Analytics System")
st.caption("Data quality, exploratory analysis, SLA breach prediction, and system use-case diagram")

REQUIRED_COLUMNS = [
    "Request_ID", "Service_Type", "Requester_Type", "Requested_Channel",
    "Priority", "Assigned_Office", "Status", "Submission_Date",
    "Queue_Length", "Processing_Days", "SLA_Days",
    "Required_Documents_Complete"
]
CATEGORY_COLUMNS = [
    "Service_Type", "Requester_Type", "Requested_Channel",
    "Priority", "Assigned_Office", "Status"
]
FEATURES = [
    "Service_Type", "Requester_Type", "Requested_Channel", "Priority",
    "Assigned_Office", "Queue_Length", "Required_Documents_Complete",
    "SLA_Days", "Submission_Weekday"
]
CAT_FEATURES = [
    "Service_Type", "Requester_Type", "Requested_Channel", "Priority",
    "Assigned_Office", "Required_Documents_Complete", "Submission_Weekday"
]
NUM_FEATURES = ["Queue_Length", "SLA_Days"]


def make_clean_dataset(data):
    """Apply the cleaning and feature-engineering steps from the notebook."""
    clean = data.copy()
    if "Request_ID" in clean.columns:
        clean = clean.drop_duplicates(subset=["Request_ID"], keep="first")

    if "Queue_Length" in clean.columns:
        clean["Queue_Length"] = pd.to_numeric(clean["Queue_Length"], errors="coerce")
        clean["Queue_Length"] = clean["Queue_Length"].abs()

    clean["Service_Type"] = clean["Service_Type"].replace({"REC_RECORDS": "Records"})
    clean["Submission_Date"] = pd.to_datetime(clean["Submission_Date"], errors="coerce")
    clean["Submission_Weekday"] = clean["Submission_Date"].dt.day_name()

    clean["Queue_Length"] = pd.to_numeric(clean["Queue_Length"], errors="coerce")
    clean["SLA_Days"] = pd.to_numeric(clean["SLA_Days"], errors="coerce")
    clean["Processing_Days"] = pd.to_numeric(clean["Processing_Days"], errors="coerce")

    clean["Queue_Category"] = pd.cut(
        clean["Queue_Length"],
        bins=[-np.inf, 5, 15, np.inf],
        labels=["Low", "Medium", "High"]
    )

    completed_mask = (
        clean["Status"].eq("Completed")
        & clean["Processing_Days"].notna()
        & clean["SLA_Days"].notna()
    )
    clean["SLA_Breached"] = np.nan
    clean.loc[completed_mask, "SLA_Breached"] = (
        clean.loc[completed_mask, "Processing_Days"]
        > clean.loc[completed_mask, "SLA_Days"]
    ).astype(int)
    return clean


def draw_use_case():
    dot = r"""
    digraph UseCase {
      graph [rankdir=LR, bgcolor="white", pad="0.3", nodesep="0.4", ranksep="0.8"];
      node [fontname="Helvetica", fontsize=10, style="filled", fillcolor="white", color="#4A5568"];
      edge [fontname="Helvetica", fontsize=9, color="#718096"];
      Student [shape=plaintext, label="Student / Requester"];
      Admin [shape=plaintext, label="Service Administrator"];
      Analytics [shape=box, style="rounded,filled", fillcolor="#EDF2F7", label="System Analytics Engine\\n(System)"];
      subgraph cluster_system {
        label="Student Service Request & Analytics System";
        style="rounded,dashed"; color="#A0AEC0"; bgcolor="#F7FAFC";
        UC1 [shape=ellipse, label="UC1: Submit Service Request"];
        UC2 [shape=ellipse, label="UC2: Track Request Status"];
        UC3 [shape=ellipse, label="UC3: Process Request"];
        UC4 [shape=ellipse, label="UC4: View SLA Performance Analytics"];
        UC5 [shape=ellipse, label="UC5: Review Delay Risk Flag"];
        UC6 [shape=ellipse, label="UC6: Calculate Breach Probability"];
        UC7 [shape=ellipse, label="UC7: Reallocate Queue Load"];
      }
      Student -> UC1; Student -> UC2;
      Admin -> UC3; Admin -> UC4; Admin -> UC5; Admin -> UC7;
      Analytics -> UC6;
      UC1 -> UC6 [label="include", style=dashed];
      UC6 -> UC5 [label="extend: Risk > Threshold", style=dashed, color="#E53E3E"];
      UC5 -> UC7 [label="extend: Manual Intervention", style=dashed, color="#DD6B20"];
    }
    """
    return dot


with st.sidebar:
    st.header("Upload dataset")
    uploaded_file = st.file_uploader(
        "Choose student_service_requests.csv",
        type=["csv"],
        help="Upload the CSV dataset used by the original notebook."
    )
    st.markdown("---")
    st.caption("Expected file: student_service_requests.csv")

if uploaded_file is None:
    st.info("Upload your `student_service_requests.csv` file in the sidebar to start the analysis.")
    st.subheader("System Use-Case Diagram")
    st.graphviz_chart(draw_use_case(), use_container_width=True)
    st.subheader("Expected dataset columns")
    st.write(", ".join(REQUIRED_COLUMNS))
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Unable to read the uploaded CSV file: {exc}")
    st.stop()

missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
if missing:
    st.error("The uploaded dataset is missing required columns:")
    st.write(missing)
    st.stop()

st.success("Dataset loaded successfully.")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Part I · Dataset", "Part II · Data Quality",
    "Part III · EDA", "Part IV · Prediction", "Use-Case Diagram"
])

with tab1:
    st.header("Part I: Dataset Loading and Understanding")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Duplicate Request IDs", f"{df['Request_ID'].duplicated().sum():,}")
    st.subheader("First 10 records")
    st.dataframe(df.head(10), use_container_width=True)
    st.subheader("Column data types")
    dtype_df = pd.DataFrame({"Column": df.columns, "Data Type": df.dtypes.astype(str).values})
    st.dataframe(dtype_df, use_container_width=True)
    st.subheader("Category frequencies")
    selected_category = st.selectbox("Select a category", CATEGORY_COLUMNS)
    freq = df[selected_category].value_counts(dropna=False).rename_axis(selected_category).reset_index(name="Count")
    st.dataframe(freq, use_container_width=True)
    st.bar_chart(freq.set_index(selected_category)["Count"])

with tab2:
    st.header("Part II: Data Quality and Feature Engineering")
    df_clean = make_clean_dataset(df)
    st.write("Cleaning steps applied:")
    st.markdown("""
    - Remove duplicate records using `Request_ID` (keep the first record).
    - Convert negative `Queue_Length` values to positive values.
    - Rename `REC_RECORDS` to `Records` in `Service_Type`.
    - Convert `Submission_Date` to a date and derive `Submission_Weekday`.
    - Create `Queue_Category`: Low (5 or less), Medium (6–15), and High (over 15).
    - Calculate `SLA_Breached` for completed requests with valid processing and SLA days.
    """)
    c1, c2, c3 = st.columns(3)
    c1.metric("Original rows", f"{len(df):,}")
    c2.metric("Cleaned rows", f"{len(df_clean):,}")
    c3.metric("Duplicates removed", f"{len(df) - len(df_clean):,}")
    quality = pd.DataFrame({
        "Column": df_clean.columns,
        "Missing Values": df_clean.isna().sum().values,
        "Missing (%)": (df_clean.isna().mean().values * 100).round(2)
    })
    st.subheader("Data quality report")
    st.dataframe(quality, use_container_width=True)
    st.subheader("Cleaned dataset preview")
    st.dataframe(df_clean.head(20), use_container_width=True)
    csv_bytes = df_clean.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download cleaned_student_service_requests.csv",
        data=csv_bytes,
        file_name="cleaned_student_service_requests.csv",
        mime="text/csv"
    )

with tab3:
    st.header("Part III: Exploratory Data Analysis")
    df_clean = make_clean_dataset(df)
    completed_reqs = df_clean[df_clean["SLA_Breached"].notna()].copy()
    if df_clean["Service_Type"].dropna().empty:
        st.warning("There are no valid Service_Type values to analyze.")
    else:
        most_requests_service = df_clean["Service_Type"].value_counts().idxmax()
        st.metric("Service type with most requests", str(most_requests_service))

    if completed_reqs.empty:
        st.warning("No completed requests with valid Processing_Days and SLA_Days were found.")
    else:
        office_breach = completed_reqs.groupby("Assigned_Office", dropna=False)["SLA_Breached"].mean().sort_values(ascending=False)
        queue_breach = completed_reqs.groupby("Queue_Category", observed=False)["SLA_Breached"].mean()
        left, right = st.columns(2)
        with left:
            st.subheader("SLA Breach Rate by Assigned Office")
            st.bar_chart(office_breach)
            st.dataframe(office_breach.rename("Breach Rate").reset_index(), use_container_width=True)
        with right:
            st.subheader("SLA Breach Rate by Queue Category")
            st.bar_chart(queue_breach)
            st.dataframe(queue_breach.rename("Breach Rate").reset_index(), use_container_width=True)
        st.caption("Breach rate is the mean of SLA_Breached (1 = breached, 0 = not breached).")

with tab4:
    st.header("Part IV: Predictive Modeling")
    df_clean = make_clean_dataset(df)
    model_df = df_clean[df_clean["SLA_Breached"].notna()].copy()
    model_df = model_df.dropna(subset=FEATURES + ["SLA_Breached"])
    if model_df.empty:
        st.warning("There are not enough complete records to train the model.")
    elif model_df["SLA_Breached"].nunique() < 2:
        st.warning("The model needs both SLA_Breached classes (0 and 1) in the completed records.")
    elif model_df["SLA_Breached"].value_counts().min() < 2:
        st.warning("There must be at least two records in each SLA_Breached class for a stratified train/test split.")
    else:
        X = model_df[FEATURES]
        y = model_df["SLA_Breached"].astype(int)
        test_size = st.slider("Test data proportion", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            preprocessor = ColumnTransformer(transformers=[
                ("num", StandardScaler(), NUM_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES)
            ])
            clf = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", DecisionTreeClassifier(max_depth=4, random_state=42))
            ])
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
            m2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.3f}")
            m3.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.3f}")
            m4.metric("F1-score", f"{f1_score(y_test, y_pred, zero_division=0):.3f}")

            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Not Breached (0)", "Breached (1)"],
                yticklabels=["Not Breached (0)", "Breached (1)"], ax=ax
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close(fig)
            st.caption("The Decision Tree uses max_depth=4 and random_state=42, matching the notebook.")
        except ValueError as exc:
            st.error(f"Model training could not be completed: {exc}")

with tab5:
    st.header("Student Service Request & Analytics System")
    st.graphviz_chart(draw_use_case(), use_container_width=True)
