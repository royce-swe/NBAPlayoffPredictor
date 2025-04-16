import pandas as pd

# Define the correct column names
col_names = ["Year", "Round", "Team 1", "Team 1 Wins", "Team 2", "Team 2 Wins", "Winner"]

# Read the CSV file that doesn't have headers (header=None tells pandas to treat all rows as data)
df = pd.read_csv("playoff_series.csv", header=None, names=col_names)

# Optionally, print the first few rows to verify
print(df.head())

# Save the updated dataframe back to a CSV file
df.to_csv("playoff_series_fixed.csv", index=False)

print("CSV with headers saved as playoff_series_fixed.csv")
