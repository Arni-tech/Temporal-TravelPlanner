import pandas as pd

clean_gap_path = r"C:\Users\negia\trip_plan\dwell_model\adjusted_dwell_model_pipeline\outputs\clean_adjusted_gap_observations.csv"
features_path = r"C:\Users\negia\trip_plan\dwell_model\adjusted_dwell_model_pipeline\outputs\final_poi_features.csv"

gap = pd.read_csv(clean_gap_path)
features = pd.read_csv(features_path)

print("Clean gap columns:")
print(gap.columns.tolist())

print("\nFeature columns:")
print(features.columns.tolist())

common_cols = sorted(set(gap.columns).intersection(set(features.columns)))
print("\nCommon columns:")
print(common_cols)