# Student Service Request & Analytics System (Streamlit)

This project converts the supplied `midterm_examination.ipynb` notebook into a Streamlit web application.

## Files

- `app.py` — Streamlit application.
- `requirements.txt` — Python package dependencies.
- `README.md` — setup and usage instructions.

## Features

1. **Part I – Dataset Loading and Understanding**
   - Upload and preview the CSV dataset.
   - View dataset size, data types, and category frequencies.
2. **Part II – Data Quality and Feature Engineering**
   - Remove duplicate `Request_ID` values.
   - Correct negative queue lengths and the `REC_RECORDS` category.
   - Create weekday, queue category, and `SLA_Breached` fields.
   - Review missing values and download the cleaned CSV.
3. **Part III – Exploratory Data Analysis**
   - Identify the service type with the most requests.
   - Compare SLA breach rates by assigned office and queue category.
4. **Part IV – Predictive Modeling**
   - Train a Decision Tree classifier.
   - Display accuracy, precision, recall, F1-score, classification report, and confusion matrix.
5. **Use-Case Diagram**
   - Display the system use-case diagram from the notebook.

## Requirements

- Python 3.9 or newer
- A CSV dataset named `student_service_requests.csv` (upload it in the app)

The CSV must contain these columns:

```text
Request_ID
Service_Type
Requester_Type
Requested_Channel
Priority
Assigned_Office
Status
Submission_Date
Queue_Length
Processing_Days
SLA_Days
Required_Documents_Complete
```

## Installation

Open a terminal in this project folder and create a virtual environment (recommended).

### Windows

```bash
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run the application

```bash
streamlit run app.py
```

Streamlit will open the application in your browser. Upload `student_service_requests.csv` using the uploader in the sidebar.

## Notes

- The notebook reads `student_service_requests.csv` from its working directory. In this Streamlit version, the CSV is uploaded through the sidebar instead.
- The application validates the required columns before running the analysis.
- SLA breach prediction is performed only on completed requests with valid processing and SLA days.
- The model uses a stratified 80/20 train-test split by default, a `StandardScaler` for numeric fields, `OneHotEncoder` for categorical fields, and a Decision Tree with `max_depth=4`.
