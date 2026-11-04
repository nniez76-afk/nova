# Campus Facility Reservation Data Mining & Analysis

Streamlit application based on the campus facility reservation analysis notebook.

## Files
- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies
- `README.md` — setup and usage guide

## Features
- Use a generated sample dataset or upload your own CSV.
- Accepts the original 9-column reservation CSV and the 12-column cleaned CSV.
- Handles common CSV encodings.
- Removes duplicate reservation IDs and handles missing/invalid values.
- Creates `Lead_Time_Category`, `Reservation_Outcome`, and `Attendance_Ratio`.
- Shows dataset summaries, EDA tables, and charts.
- Trains a Decision Tree and reports accuracy, precision, recall, F1-score, confusion matrix, classification report, and feature importance.
- Downloads raw and cleaned datasets.

## Required CSV columns

The uploaded CSV must contain these columns (exact spelling and capitalization):

1. `Reservation_ID`
2. `User_Category`
3. `Facility_Type`
4. `Lead_Time_Days`
5. `Participants`
6. `Actual_Attendance`
7. `Usage_Status`
8. `Department`
9. `Time_Slot`

The app also accepts the three derived columns from an already-cleaned file:
`Lead_Time_Category`, `Reservation_Outcome`, and `Attendance_Ratio`.
These are recalculated from the base columns so the values stay consistent.

## Run locally

Install Python 3.9 or newer. Open a terminal in this folder and run:

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

macOS/Linux:
```bash
source venv/bin/activate
```

Install packages:
```bash
python -m pip install -r requirements.txt
```

Start the app:
```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

## Deploy with Streamlit Community Cloud

1. Upload `app.py`, `requirements.txt`, and `README.md` to the same GitHub repository folder.
2. In Streamlit Community Cloud, create or open the app and select that repository and `app.py`.
3. Deploy or reboot the app.
4. If the browser shows a stale JavaScript/module error, hard-refresh with `Ctrl + Shift + R` or open the app in a private window. If it continues, check the app's Cloud logs and share the error details.

## Model notes

The Decision Tree uses `User_Category`, `Facility_Type`, `Lead_Time_Days`, `Department`, `Time_Slot`, and `Participants` as predictors. `Usage_Status` defines the target and is not used as an input; `Actual_Attendance` is also excluded to reduce data leakage. The test split is 20%, stratified, with `random_state=42` and `max_depth=4`.

Model results depend on the uploaded dataset and are not guarantees of future outcomes.
