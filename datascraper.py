import pandas as pd
import time
from nba_api.stats.endpoints import LeagueDashTeamStats
from nba_api.stats.static import teams as nba_teams_static
from nba_api.stats.endpoints import TeamInfoCommon


# Load your playoff series CSV with headers
series_df = pd.read_csv("playoff_series_fixed.csv")

# Clean team name columns: remove extra spaces
series_df["Team 1"] = series_df["Team 1"].str.strip()
series_df["Team 2"] = series_df["Team 2"].str.strip()

# Get unique seasons from your dataset (e.g., 2015-2024)
seasons = series_df["Year"].unique()

# Create a dictionary to store team stats for each season
# The key is a tuple: (numeric season, team name)
team_stats_dict = {}

# Helper function to convert numeric season to NBA season string.
# For example, 2015 becomes "2014-15", 2024 becomes "2023-24"
def convert_season(year):
    year = int(year)
    return f"{year-1}-{str(year)[-2:]}"

# Loop through each season and fetch team stats once using the NBA API
for season in seasons:
    season_str = convert_season(season)
    print(f"Fetching stats for season {season_str} (CSV year {season})...")
    try:
        # Fetch the team stats for the season using LeagueDashTeamStats
        stats_df = LeagueDashTeamStats(season=season_str).get_data_frames()[0]
        
        # Iterate through the rows of the stats DataFrame and store in our dictionary
        for idx, row in stats_df.iterrows():
            # Normalize the team name (the API returns it in the 'TEAM_NAME' column)
            team_name = row["TEAM_NAME"].strip()
            team_stats_dict[(season, team_name)] = {
                "FG_PCT": row["FG_PCT"],
                "FG3_PCT": row["FG3_PCT"],
                "PTS": row["PTS"],
                "WinPCT": row["W_PCT"],
            }
    except Exception as e:
        print(f"Error fetching data for season {season_str}: {e}")
    
    # Pause to respect API rate limits
    time.sleep(2)

# Check that all teams from your CSV are present in the fetched stats
for season in seasons:
    # Get a unique set of teams appearing in this season from both Team 1 and Team 2
    teams_in_csv = set(
        series_df.loc[series_df["Year"] == season, "Team 1"].tolist() +
        series_df.loc[series_df["Year"] == season, "Team 2"].tolist()
    )
    for team in teams_in_csv:
        if (season, team) not in team_stats_dict:
            print(f"Warning: Stats not found for team '{team}' in season {season}")

# Define a helper function to get stats from our dictionary.
def get_team_stats(season, team_name):
    # Clean the team name
    clean_team_name = team_name.strip()
    # For the 2015 season, override "LA Clippers" with "Los Angeles Clippers"
    if season == 2015 and clean_team_name == "LA Clippers":
        clean_team_name = "Los Angeles Clippers"
    
    key = (season, clean_team_name)
    if key in team_stats_dict:
        return team_stats_dict[key]
    else:
        print(f"Stats not found for {clean_team_name} in {season}")
        return {"FG_PCT": None, "FG3_PCT": None, "PTS": None, "WinPCT": None}

# Function to add stats columns for a given team column (e.g., "Team 1" or "Team 2")
def add_stats(row, team_column_prefix):
    season = row["Year"]
    team_name = row[team_column_prefix]
    stats = get_team_stats(season, team_name)
    row[f"{team_column_prefix}_FG_PCT"] = stats["FG_PCT"]
    row[f"{team_column_prefix}_FG3_PCT"] = stats["FG3_PCT"]
    row[f"{team_column_prefix}_PTS"] = stats["PTS"]
    row[f"{team_column_prefix}_WinPCT"] = stats["WinPCT"]
    return row

# Apply the helper function for both Team 1 and Team 2 columns
series_df = series_df.apply(lambda row: add_stats(row, "Team 1"), axis=1)
series_df = series_df.apply(lambda row: add_stats(row, "Team 2"), axis=1)

# Save the enriched dataset to a new CSV file
output_csv = "playoff_series_with_stats.csv"
series_df.to_csv(output_csv, index=False)
print(f"Merged data saved to {output_csv}")


# ---------------------------
# PART 2: Create a CSV file for the top 8 teams in the East and West (latest season)
# ---------------------------


# Retrieve the list of all NBA teams
teams_list = nba_teams_static.get_teams()

# Initialize a dictionary to map team full names to their respective conferences
conference_mapping = {}

# Iterate over each team to fetch its conference information
for team in teams_list:
    team_id = team['id']
    team_name = team['full_name'].strip()

    try:
        # Fetch team details using the TeamInfoCommon endpoint
        team_info = TeamInfoCommon(team_id=team_id)
        team_details = team_info.get_data_frames()[0]
        conference = team_details.loc[0, 'TEAM_CONFERENCE']
        conference_mapping[team_name] = conference
    except Exception as e:
        print(f"Error fetching data for team {team_name}: {e}")

    # Pause to respect API rate limits
    time.sleep(1)

# Determine the latest season from your data
latest_season = max(seasons)
print(f"Processing top 8 teams for season {latest_season}...")

# Collect team stats for the latest season
latest_team_stats = []
for (season, team_name), stats in team_stats_dict.items():
    if season == latest_season:
        conf = conference_mapping.get(team_name, None)
        if conf is None:
            print(f"Warning: Conference not found for team '{team_name}'")
        latest_team_stats.append({
            "Team": team_name,
            "Conference": conf,
            "FG_PCT": stats["FG_PCT"],
            "FG3_PCT": stats["FG3_PCT"],
            "PTS": stats["PTS"],
            "WinPCT": stats["WinPCT"]
        })

# Create a DataFrame from the latest season's stats
df_latest = pd.DataFrame(latest_team_stats)

# Filter teams by conference
df_east = df_latest[df_latest["Conference"] == "East"]
df_west = df_latest[df_latest["Conference"] == "West"]

# Sort by winning percentage and select top 8 teams from each conference
df_east_top8 = df_east.sort_values(by="WinPCT", ascending=False).head(8)
df_west_top8 = df_west.sort_values(by="WinPCT", ascending=False).head(8)

# Combine the top teams from both conferences
df_top8 = pd.concat([df_east_top8, df_west_top8], ignore_index=True)

# Rename "WinPCT" to "PCT" for consistency
df_top8 = df_top8.rename(columns={"WinPCT": "PCT"})

# Rearrange columns for clarity
df_top8 = df_top8[["Team", "Conference", "FG_PCT", "FG3_PCT", "PTS", "PCT"]]

# Save the DataFrame to a CSV file
df_top8.to_csv("top8_teams_by_conference.csv", index=False)
print("CSV file 'top8_teams_by_conference.csv' created.")