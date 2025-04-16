
# NBA Playoff Prediction AI

This project predicts the outcomes of NBA playoff series using a machine learning model. The model uses historical data, team statistics, and a Random Forest Regressor to predict the probability of a team winning a playoff series. It also simulates an entire playoff bracket, including the Play-In Tournament, and determines the final NBA Champion. 

I did some small datascraping so I included that code in datascraper.py if you want to view it or use it. I likely would've done more but the basketball reference website I was using was formatted weirdly and made it hard to scrape all the data off the website. I also performed some small file fixing so that is included in fileFixer.py

## Notable Limitations

While this project makes a good attempt to predict the playoff series of the 2025 NBA Season, it relies heavily on WinPCT. This results in all of the series being decided on which team has the higher WinPCT and is relatively uninfluenced by other stats. This is likely due to the fact all that previous playoff data obviously displays higher seeded teams performing better during the post season.

## Future Updates

In the future I will likely include player stats/data in an attempt to deter the AI's extreme reliance on team seeding.

## Requirements

To run this project, you'll need to have the following Python libraries installed:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

You can install them using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Project Structure

### Files:
- `playoff_series_with_stats.csv`: Historical playoff series data, including team statistics such as FG%, 3P%, PTS, and Win%.
- `2025 NBA Teams Stats - Sheet2.csv`: Current season (2025) team statistics.
- `main.py`: The main Python script for training the model and simulating playoff matchups.

### Code Overview

1. **Data Preparation**:
    - Historical playoff data is loaded, and feature differences between teams (FG%, FG3%, PTS, WinPCT, SRS) are computed.
    - The target variable `p` represents the probability of team 1 winning the series based on their performance in the regular season.

2. **Model Training**:
    - A Random Forest Regressor is trained on the historical data to predict the probability of team 1 winning a playoff series based on their stats relative to the opponent.

3. **Playoff Simulation**:
    - A function `simulate_series` simulates a best-of-7 playoff series between two teams using the trained model. Noise is added to the game win probability to reflect randomness.
    - The Play-In Tournament is simulated to determine the 7th and 8th seeds for each conference.

4. **Bracket Simulation**:
    - The main function `simulate_bracket` simulates an entire conference's playoff bracket, including all rounds until the final.
    - The NBA Finals are simulated by determining the winner of the Eastern and Western Conference finals.

## Usage

To run the full simulation, execute the following code in `nba_playoff_prediction.py`:

```python
# Run Play-In for both conferences
east_playoff_seeds = play_in_tournament(east_seeds, model, "Eastern")
west_playoff_seeds = play_in_tournament(west_seeds, model, "Western")

# Run Conference Brackets
east_champion = simulate_bracket(east_playoff_seeds, model, "Eastern")
west_champion = simulate_bracket(west_playoff_seeds, model, "Western")

# Run NBA Finals
winner, east_win_prob, west_win_prob, avg_wins_east, avg_wins_west = simulate_series(east_champion[0], west_champion[0], model)

print(f"{east_team} vs {west_team} → NBA Champion: {final_winner['Team']}")
print(f"Avg Wins: {east_team} {avg_wins_east:.2f} - {west_team} {avg_wins_west:.2f}")
```

This will simulate the Play-In Tournament, conference brackets, and NBA Finals, and output the winner of the NBA Championship.

## Notes

- The model uses historical data to predict win probabilities, but real-life results can be influenced by many factors such as injuries, player performance on the day, and other variables not captured in the data.
- The simulation incorporates randomness to account for variability in the outcome of individual games within a series.

## License

This project is licensed under the MIT License.
