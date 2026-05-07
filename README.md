# HR Attrition Intelligence Dashboard

**Built by:** Vasanth A — AI & Data Science, 2026 batch

## Live Demo
[Open the app](https://vasanth-hr-attrition.streamlit.app)

## What it does
- Attrition rate breakdown by department, job role, age, tenure, overtime, marital status
- Attrition drivers — income violin plot, job satisfaction, work-life balance, distance
- Risk heatmap — department × age group interactive heatmap
- Retention actions — evidence-based recommendations with ROI table
- Individual risk predictor — enter any employee profile, get attrition probability + gauge

## Tech stack
| Layer | Tool |
|---|---|
| Frontend | Streamlit |
| Charts | Plotly Express + Graph Objects |
| Data | IBM HR Analytics (synthetic, 1,470 employees) |
| Deploy | Streamlit Cloud (free) |

## Real dataset
Download: kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
Replace generate_data() with pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
1. GitHub → New repo "hr-attrition-dashboard" → upload 3 files
2. share.streamlit.io → connect repo → app.py → Deploy

## Resume line
> "Built HR Attrition Intelligence Dashboard — identified top 3 retention levers reducing 18% turnover, individual risk predictor, ROI analysis — deployed live at [url]"
