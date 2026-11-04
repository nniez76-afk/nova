# Campus Facility Reservation Data Mining & Analysis

A Streamlit web application converted from the original `NOVAFINAL.ipynb` notebook.

The application supports sample data or a user-uploaded CSV, data cleaning, feature engineering, exploratory data analysis, visualizations, and Decision Tree classification.

## Features

- Generate the sample campus facility reservation dataset (400 records).
- Upload a CSV file.
- Read common CSV encodings (`utf-8-sig`, `utf-8`, `cp1252`, and `latin1`).
- Validate required columns.
- Remove duplicate reservation IDs.
- Normalize user categories and handle missing values.
- Correct invalid lead-time and attendance values.
- Create:
  - `Lead_Time_Category`
  - `Reservation_Outcome`
  - `Attendance_Ratio`
- Display dataset summaries and previews.
- Visualize reservation counts and outcomes.
- Train and evaluate a Decision Tree classifier.
- Display accuracy, precision, recall, F1-score, confusion matrix, classification report, and feature importance.
- Download raw and cleaned datasets as CSV files.

## Project Files

```text
campus_facility_streamlit/
├── app.py
├── README.md
└── requirements.txt
```

## Requirements

- Python 3.9 or newer
- pip

## Installation

1. Download or extract the project folder.
2. Open a terminal or Command Prompt in the project folder.
3. (Recommended) Create and activate a virtual environment.

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

4. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run the Streamlit App

Run this command inside the project folder:

```bash
streamlit run app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open that address in your browser.

## Uploading Your Own CSV

Choose **Upload my CSV file** in the sidebar.

Your CSV must contain these exact column names:

| Column | Description |
|---|---|
| `Reservation_ID` | Unique reservation identifier |
| `User_Category` | Student, Faculty, Staff, or another user category |
| `Facility_Type` | Type of campus facility |
| `Lead_Time_Days` | Number of days between reservation and use |
| `Participants` | Number of expected participants |
| `Actual_Attendance` | Number of people who attended |
| `Usage_Status` | Used, Cancelled, or No-Show |
| `Department` | User's department |
| `Time_Slot` | Morning, Afternoon, or Evening |

Column names are case-sensitive. Rename your CSV columns to match the table before uploading if needed.

## Notes About the Model

The Decision Tree uses these predictors, which are intended to be available when a reservation is submitted:

- User_Category
- Facility_Type
- Lead_Time_Days
- Department
- Time_Slot
- Participants

`Usage_Status` is used to create the target (`Reservation_Outcome`), but it is not used as a model predictor. `Actual_Attendance` is also excluded from the predictors to help prevent data leakage.

The target classes are:

- `Completed` = 0
- `Not Completed` = 1

The model uses an 80% training and 20% testing split, stratified by the target, with `random_state=42`. The Decision Tree has `max_depth=4`.

Model scores depend on the dataset and should not be treated as guaranteed predictions of future reservations.

## Troubleshooting

### `streamlit` is not recognized

Make sure your virtual environment is activated, then run:

```bash
python -m pip install -r requirements.txt
```

You can also launch it with:

```bash
python -m streamlit run app.py
```

### CSV encoding error

Open the CSV in Excel or another spreadsheet application and save it as **CSV UTF-8**, then upload it again. The app also tries several common encodings automatically.

### Missing required columns

Check that your CSV contains all nine required columns listed above and that the spelling and capitalization match exactly.
