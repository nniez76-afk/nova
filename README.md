# SLA Breach Decision-Support System (Streamlit)

Deploys the **Midterm Laboratory Examination - Section B (Student Service Request Monitoring)** as a Streamlit
app: it predicts whether a student service request will breach its SLA and supports staff review. Predictions are
**decision support only** - they never authorise denying or delaying a service.

| Page | What it does |
|---|---|
| Home | Overview, how-to, decision-support notice |
| Dataset | CSV upload, schema checks, preview, null / duplicate summary, midterm cleaning report |
| Data Analysis | Descriptive stats, class distribution, 5 interactive Plotly charts |
| Prediction | Single request (5 sample scenarios), batch scoring, metrics, consistency check, limitations |
| Dashboard | KPIs, risk/review queue, interpretation, recorded human decisions (role-based) |
| System Design | Requirements, actors, updated use case diagram, workflow, traceability |
| Midterm Notebook | Read-only view of the submitted `.ipynb` + live re-run of Parts II-IV |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## IMPORTANT - use your real midterm data
The repo ships a **synthetic** sample (`data/student_service_requests.csv`, same 14-column schema) so it can be public.
To reproduce your exact midterm model and numbers:
```bash
# 1. put your real CSV at data/student_service_requests.csv (only if authorised to publish it)
python train_model.py            # same Part II cleaning + Part IV pipeline, saves models/*.joblib + metadata
# 2. commit models/trained_model.joblib and models/model_metadata.json
```
If the real data is private, keep it out of Git and upload it on the **Dataset** page instead; train once locally and commit only the model.

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repository (never commit passwords or student records).
2. Go to https://share.streamlit.io -> sign in -> **Create app**.
3. Pick the repository and branch, set **Main file path** to `app.py`, click **Deploy**.
4. If the build fails, open **Manage app -> Logs**.

If the saved joblib was created with a different scikit-learn version than the host, the app detects it and rebuilds
the identical pipeline once from the bundled data (a notice is shown). Re-run `python train_model.py` with the
host's version to avoid this.

## Project structure
```
app.py                        navigation, caching, roles
requirements.txt
train_model.py                offline training (midterm Part II + IV)
modules/data_processing.py    upload, validation, cleaning
modules/exploratory_analysis.py
modules/prediction.py         inference, validation, metrics
modules/dashboard.py          KPIs, review queue, decisions
modules/system_design.py      SAD page
modules/model_training.py     midterm pipeline definition
modules/midterm_notebook.py   notebook viewer + live re-run
models/trained_model.joblib   saved preprocessing + model pipeline
models/model_metadata.json    version, timestamp, metrics, test cases
assets/use_case_diagram.png   updated use case diagram (+ midterm version, workflow)
notebooks/midterm_examination.ipynb
data/student_service_requests.csv   synthetic sample
scripts/                      sample-data and diagram generators
tests/test_pipeline.py        pytest suite
DEPLOYMENT_REPORT.md
```

## Tests
`pytest -q` runs 18 checks: schema validation, cleaning rules, no-leakage, valid/invalid inputs,
batch scoring, and reproduction of the stored test predictions.
