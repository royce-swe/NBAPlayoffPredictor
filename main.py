import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import warnings
from sklearn.exceptions import DataConversionWarning

# Suppress warnings about feature names and data conversion
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)

# -----------------------------
# PART 1: Data Preparation and Model Training
# -----------------------------

# Load the historical playoff series data.
# Expected headers: Year, Round, Team 1, Team 1 Wins, Team 2, Team 2 Wins, Winner,
# Team 1_FG_PCT, Team 1_FG3_PCT, Team 1_PTS, Team 1_WinPCT,
# Team 2_FG_PCT, Team 2_FG3_PCT, Team 2_PTS, Team 2_WinPCT
data = pd.read_csv("playoff_series_with_stats.csv")
data.columns = data.columns.str.strip()
team_stats_2025 = pd.read_csv('2025 NBA Teams Stats - Sheet2.csv')

east_teams = team_stats_2025[team_stats_2025['Conference'] == 'East']
west_teams = team_stats_2025[team_stats_2025['Conference'] == 'West']

east_seeds = east_teams.sort_values(by='PCT', ascending=False).reset_index(drop=True)
west_seeds = west_teams.sort_values(by='PCT', ascending=False).reset_index(drop=True)

east_matchups = [
    (east_seeds.iloc[0].to_dict(), east_seeds.iloc[7].to_dict()),
    (east_seeds.iloc[1].to_dict(), east_seeds.iloc[6].to_dict()),
    (east_seeds.iloc[2].to_dict(), east_seeds.iloc[5].to_dict()),
    (east_seeds.iloc[3].to_dict(), east_seeds.iloc[4].to_dict()),
]


west_matchups = [
    (west_seeds.iloc[0].to_dict(), west_seeds.iloc[7].to_dict()),
    (west_seeds.iloc[1].to_dict(), west_seeds.iloc[6].to_dict()),
    (west_seeds.iloc[2].to_dict(), west_seeds.iloc[5].to_dict()),
    (west_seeds.iloc[3].to_dict(), west_seeds.iloc[4].to_dict()),
]


# Create feature differences: subtract Team 2 stats from Team 1 stats
data["diff_FG_PCT"]   = data["Team 1_FG_PCT"] - data["Team 2_FG_PCT"]
data["diff_FG3_PCT"]  = data["Team 1_FG3_PCT"] - data["Team 2_FG3_PCT"]
data["diff_PTS"]      = data["Team 1_PTS"] - data["Team 2_PTS"]
data["diff_WinPCT"]   = data["Team 1_WinPCT"] - data["Team 2_WinPCT"]
data["diff_SRS"] = data["Team 1_SRS"] - data["Team 2_SRS"]

# Construct a target value 'p' representing team1's game win probability.
# For a best-of-7 series, the winning team always wins 4 games.
# We'll compute p as follows:
# If team1 is the winner (Team 1 Wins == 4): p = 4 / (4 + Team 2 Wins)
# Otherwise (Team 2 is the winner): p = Team 1 Wins / (Team 1 Wins + 4)
def calc_game_win_prob(row):
    if row["Team 1 Wins"] == 4:
        return 4 / (4 + row["Team 2 Wins"])
    else:
        return row["Team 1 Wins"] / (row["Team 1 Wins"] + 4)

data["p"] = data.apply(calc_game_win_prob, axis=1)

###Features and target
features = ["diff_FG_PCT", "diff_FG3_PCT", "diff_PTS", "diff_WinPCT", "diff_SRS"]
X = data[features]
y = data["p"]

# Split data for evaluation (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a regression model to predict game win probability for team1 in a series.
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model performance (optional)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model MSE: {mse:.4f}")

# -----------------------------
# PART 3: Predicting a 2025 Playoff Matchup
# -----------------------------
# Example: Suppose you have two teams for a matchup.
# You need to supply their current season statistics (FG%, FG3%, PTS, WinPCT).
# For this example, we'll define dummy stats for team1 and team2.
# In practice, you'd load your 2025 data similarly to how you loaded historical data.

# Example stats for two hypothetical teams (for 2025):
# (These values should come from your 2025 data; here they are placeholders.)

def simulate_series(team1, team2, model, n_simulations=1000):
    """
    Simulate a best-of-7 series between team1 and team2 using the model's game win probability.
    Adds extra noise in each game simulation to represent game-by-game randomness.
    Returns:
      winner (dict): Team dictionary for the series winner.
      team1_win_prob (float): Estimated series win probability for team1.
      team2_win_prob (float): Estimated series win probability for team2.
      avg_wins1 (float): Average games won by team1 in the series.
      avg_wins2 (float): Average games won by team2 in the series.
    """
    # Compute the base win probability (p) from the model:
    features = np.array([
        team1['FG_PCT'] - team2['FG_PCT'],
        team1['FG3_PCT'] - team2['FG3_PCT'],
        team1['PTS'] - team2['PTS'],
        team1['PCT'] - team2['PCT'],
        team1['SRS'] - team2['SRS']
    ]).reshape(1, -1)
    
    p = model.predict(features)[0]
    
    # For each simulation, add additional noise to p (using a larger standard deviation)
    # and optionally clip the value to a more competitive range (for close matchups).
    team1_series_wins = 0
    team2_series_wins = 0
    team1_games_won = []
    team2_games_won = []
    
    for _ in range(n_simulations):
        # For each simulated series, add extra noise to the base probability:
        p_noisy = np.clip(p + np.random.normal(0, 0.05), 0.0, 1.0)
        
        wins1, wins2 = 0, 0
        while wins1 < 4 and wins2 < 4:
            # For each game, simulate win using the noisy probability.
            if np.random.rand() < p_noisy:
                wins1 += 1
            else:
                wins2 += 1
        team1_games_won.append(wins1)
        team2_games_won.append(wins2)
        if wins1 == 4:
            team1_series_wins += 1
        else:
            team2_series_wins += 1
    
    # Calculate average wins across all simulations:
    avg_wins1 = min(round(np.mean(team1_games_won), 1), 4)
    avg_wins2 = min(round(np.mean(team2_games_won), 1), 4)
    
    team1_win_prob = team1_series_wins / n_simulations
    team2_win_prob = team2_series_wins / n_simulations
    
    print(f"Series Prediction: {avg_wins1}-{avg_wins2} (Team1 win probability: {team1_win_prob * 100:.2f}%)")
    
    # Determine the series winner by overall simulation probability:
    winner = team1 if team1_win_prob > team2_win_prob else team2
    
    return winner, team1_win_prob, team2_win_prob, avg_wins1, avg_wins2





# -----------------------------
# PART 4: Bracket Simulation
# -----------------------------


def play_in_tournament(seeds, model, conference_name=""):
    """
    Simulate the Play-In Tournament to determine 7th and 8th seeds.
    seeds: DataFrame sorted by win percentage (seeds 1-10)
    Returns updated seeds with correct 7 and 8 seeds for the playoffs.
    """
    print(f"\n🏀 {conference_name} Play-In Tournament")
    
    # Seeds 7-10
    team7 = seeds.iloc[6].to_dict()
    team8 = seeds.iloc[7].to_dict()
    team9 = seeds.iloc[8].to_dict()
    team10 = seeds.iloc[9].to_dict()

    # Game 1: 7 vs 8 → Winner gets 7th seed
    _, team7_win_prob, _, _, _ = simulate_series(team7, team8, model)
    winner78 = team7 if team7_win_prob > 0.5 else team8
    loser78 = team8 if team7_win_prob > 0.5 else team7
    print(f"{team7['Team']} vs {team8['Team']} → Winner: {winner78['Team']} (7th Seed)")

    # Game 2: 9 vs 10 → Loser is eliminated
    _, team9_win_prob, _, _, _ = simulate_series(team9, team10, model)
    winner910 = team9 if team9_win_prob > 0.5 else team10
    print(f"{team9['Team']} vs {team10['Team']} → Winner: {winner910['Team']}")

    # Game 3: Loser of 7/8 vs Winner of 9/10 → Winner gets 8th seed
    _, final_win_prob, _, _, _ = simulate_series(loser78, winner910, model)
    winner_final = loser78 if final_win_prob > 0.5 else winner910
    print(f"{loser78['Team']} vs {winner910['Team']} → Winner: {winner_final['Team']} (8th Seed)")

    # Build final 1–8 seeds
    playoff_seeds = pd.concat([
        seeds.iloc[:6],                   # Seeds 1-6
        pd.DataFrame([winner78]),         # 7th seed
        pd.DataFrame([winner_final])      # 8th seed
    ]).reset_index(drop=True)

    return playoff_seeds



def simulate_bracket(playoff_seeds, model, conference_name=""):
    """
    Simulate an entire conference playoff bracket.
    """
    # First round matchups based on seeding (1 vs 8, 2 vs 7, etc.)
    matchups = [
        (playoff_seeds.iloc[0].to_dict(), playoff_seeds.iloc[7].to_dict()),
        (playoff_seeds.iloc[1].to_dict(), playoff_seeds.iloc[6].to_dict()),
        (playoff_seeds.iloc[2].to_dict(), playoff_seeds.iloc[5].to_dict()),
        (playoff_seeds.iloc[3].to_dict(), playoff_seeds.iloc[4].to_dict())
    ]

    round_num = 1
    current_round = matchups
    while len(current_round) > 1:
        print(f"\n--- {conference_name} Round {round_num} ---")
        next_round = []
        for team1, team2 in current_round:
            winner, win1_pct, win2_pct, avg_wins1, avg_wins2 = simulate_series(team1, team2, model)
            print(f"{team1['Team']} vs {team2['Team']} → Winner: {winner['Team']} | Avg Wins: {team1['Team']} {avg_wins1:.2f} - {team2['Team']} {avg_wins2:.2f}")
            next_round.append(winner)
        current_round = [(next_round[i], next_round[i + 1]) for i in range(0, len(next_round), 2)]
        round_num += 1

    # Final round
    team1, team2 = current_round[0]
    winner, win1_pct, win2_pct, avg_wins1, avg_wins2 = simulate_series(team1, team2, model)

    print(f"\n--- {conference_name} Final ---")
    print(f"{team1['Team']} vs {team2['Team']} → Winner: {winner['Team']} | Avg Wins: {team1['Team']} {avg_wins1:.2f} - {team2['Team']} {avg_wins2:.2f}")

    print(f"\n🏆 {conference_name} Champion: {winner['Team']}")
    return winner, win1_pct, win2_pct, avg_wins1, avg_wins2





# -----------------------------
# Run Full Bracket Simulation
# -----------------------------

# 1. Run Play-In for both conferences
east_playoff_seeds = play_in_tournament(east_seeds, model, "Eastern")
west_playoff_seeds = play_in_tournament(west_seeds, model, "Western")

# 2. Run Conference Brackets
east_champion = simulate_bracket(east_playoff_seeds, model, "Eastern")
west_champion = simulate_bracket(west_playoff_seeds, model, "Western")

# 3. Run NBA Finals
print("\n====================")
print("🏀 Simulating NBA Finals")
print("====================")

# Unpack all five return values from simulate_series for the NBA Finals matchup
winner, east_win_prob, west_win_prob, avg_wins_east, avg_wins_west = simulate_series(east_champion[0], west_champion[0], model)

# Get the team names for clarity
east_team = east_champion[0]['Team']
west_team = west_champion[0]['Team']

# Determine the final winner (this is redundant since simulate_series already returns a winner)
final_winner = east_champion[0] if east_win_prob > west_win_prob else west_champion[0]

print(f"{east_team} vs {west_team} → NBA Champion: {final_winner['Team']}")
print(f"Avg Wins: {east_team} {avg_wins_east:.2f} - {west_team} {avg_wins_west:.2f}")

