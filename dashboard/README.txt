QUANTEDGE DASHBOARD UPDATE

Replace these three existing files in your project:
1) dashboard/app.py                  <- app.py
2) dashboard/components/data_loader.py <- components/data_loader.py
3) dashboard/components/charts.py      <- components/charts.py

IMPORTANT: Extract artifacts.zip into your project ROOT so that the path is:
<project-root>/artifacts/traditional_strategy_research/...
Do not place it inside dashboard/.

Expected structure:
QuantEdge/
  artifacts/
  data/raw/
  dashboard/
    app.py
    components/data_loader.py
    components/charts.py

Run from project root:
streamlit run dashboard/app.py

The dashboard uses saved CSVs. It does not retrain or alter model outputs.
