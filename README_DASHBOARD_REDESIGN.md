# QuantEdge Dashboard Redesign

This is a replacement dashboard UI that reads existing QuantEdge artifacts. It does not retrain models or modify research pipelines.

## Files
- `dashboard/app.py` — complete redesigned Streamlit application and navigation.
- `dashboard/components/data_loader.py` — artifact and market-data loading.
- `dashboard/components/charts.py` — reusable Plotly charts.

## Install / run (PowerShell)
From your project root (`C:\Users\<you>\QuantEdge`):

```powershell
pip install streamlit plotly pandas numpy
streamlit run dashboard/app.py
```

If `streamlit` is not recognized:
```powershell
python -m streamlit run dashboard/app.py
```

## Safe installation
1. Make a backup of your current `dashboard` folder (e.g. `dashboard_backup`).
2. Extract this ZIP into your QuantEdge project root, allowing the `dashboard` folder to merge/replace.
3. Run the command above.

No existing `src/`, `data/`, `artifacts/`, model files, experiment scripts, or tests are changed.

## Data behavior
- Market prices load from `data/raw` first, then `artifacts/raw`.
- Traditional, hybrid, technical ML and hybrid ML pages read their saved CSV artifacts.
- Phase 6 portfolio and Phase 7 SHAP panels appear only when their artifacts exist.
- Missing results are shown as missing; no metrics are fabricated.
- The app is historical research only, not a live market feed.

## Important
Keep the `dashboard` folder at the QuantEdge project root so the root-path calculation works.
