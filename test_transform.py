from transform import ECBDataTransformer
import pandas as pd



# Initialize
print('Initializing transformer...')
transformer = ECBDataTransformer()
# Test with specific series: FSI CISS, YC 10Y, MMR 3M, INF total, LFI unemployment
test_series = [4, 19, 12, 22, 18]

data, _, report = transformer.create_analysis_dataset(
  target_freq='ME',
  start_date='2004-09-01',
  end_date='2026-01-01',
  auto_transform=True,
  return_stationarity_report=True
)

print(f'\n✓ Pipeline completed successfully!')
data.to_parquet('data.parquet')
print(f'\nStationarity Report:')
print(report[['series_name','overall_stationary', 'recommended_transform']].to_string(index=False))
print("---Retesting Stationarity---")

fresh_report = []
for col in data.columns:
    series = data[col].dropna()
    result = transformer.test_stationarity(series, verbose=False)
    fresh_report.append(result.to_dataframe())

fresh_stationarity_report = pd.concat(fresh_report, ignore_index=True)
print(f'\nFresh Stationarity Report:')
print(fresh_stationarity_report[['series_name','overall_stationary', 'recommended_transform']].to_string(index=False))
