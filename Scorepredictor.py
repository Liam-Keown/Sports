import numpy as np
import torch
from torch import nn, optim
import requests
from bs4 import BeautifulSoup
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler

class Team:
    def __init__(self, name):
        self.name = name 
        self.no_wins = 0
        self.no_losses = 0
        self.no_draws = 0
        self.points = 0
        self.goals_scored = 0
        self.goals_conceded = 0

        self.form_wins = 0
        self.form_losses = 0
        self.form_draws = 0
        self.form_points = 0
        self.form_goals_scored = 0
        self.form_goals_conceded = 0

    def get_games(self, results, no_games):
        home_games = results[results['Home Team'] == self.name]
        away_games = results[results['Away Team'] == self.name]
        games_stacked = pd.concat([home_games, away_games])
        games = games_stacked.sort_index()
        return games.iloc[:no_games]
    
    @staticmethod
    def get_goals(data):
        score_str = data['Result']
        home_goals, away_goals = score_str.split(' - ')
        return int(home_goals), int(away_goals)
    
    def determine_and_analyse_result(self, data):
        home, away = Team.get_goals(data)
        if self.name == data['Home Team']:
            self.goals_scored += home
            self.goals_conceded += away
            if home > away:
                self.no_wins += 1
                self.points += 3
            elif away > home:
                self.no_losses += 1
            else:
                self.no_draws += 1
                self.points += 1
        elif self.name == data['Away Team']:
            self.goals_scored += away
            self.goals_conceded += home            
            if away > home:
                self.no_wins += 1
                self.points += 3
            elif home > away:
                self.no_losses += 1
            else:
                self.no_draws += 1
                self.points += 1

    def track_form(self, data):
        home, away = Team.get_goals(data)
        if self.name == data['Home Team']:
            self.form_goals_scored += home
            self.form_goals_conceded += away
            if home > away:
                self.form_wins += 1
                self.form_points += 3
            elif away > home:
                self.form_losses += 1
            else:
                self.form_draws += 1
                self.form_points += 1
        elif self.name == data['Away Team']:
            self.form_goals_scored += away
            self.form_goals_conceded += home            
            if away > home:
                self.form_wins += 1
                self.form_points += 3
            elif home > away:
                self.form_losses += 1
            else:
                self.form_draws += 1
                self.form_points += 1

def read_results(filename):
    df = pd.read_csv(filename)
    return df

results_file = 'epl-2023-GMTStandardTime.csv'
prem_season = read_results(results_file)
no_games_played = 22

team_list = read_results('teams.csv')
teams = team_list['Team']
team_scores = team_list['Score']

desired_team = 'Man Utd'
desired_team_obj = Team(desired_team)

desired_games = desired_team_obj.get_games(prem_season, no_games_played)
form_games = desired_games[-5:]

for i in range(len(desired_games)):
    desired_team_obj.determine_and_analyse_result(desired_games.iloc[i])

for i in range(5):
    desired_team_obj.track_form(form_games.iloc[i]) 

# print("SEASON: Goals:",desired_team_obj.goals_scored,"   Points:",desired_team_obj.points,"   Wins:",desired_team_obj.no_wins)
# print("FORM: Goals:",desired_team_obj.form_goals_scored,"   Points:",desired_team_obj.form_points,"   Wins:",desired_team_obj.form_wins)

#print(desired_games)


team_quality = {
    'Burnley': 60, 'Arsenal': 85, 'Bournemouth': 70, 'Brighton': 75, 'Everton': 68,
    'Sheffield Utd': 55, 'Newcastle': 80, 'Brentford': 72, 'Chelsea': 82, 'Man Utd': 78,
    'Nottingham Forest': 65, 'Fulham': 66, 'Liverpool': 90, 'Wolves': 62, 'Spurs': 77,
    'Crystal Palace': 69, 'Aston Villa': 71, 'West Ham': 74, 'Man City':94, 'Luton':53
}

# Add quality scores to the dataframe
desired_games['Home Quality'] = desired_games['Home Team'].map(team_quality)
desired_games['Away Quality'] = desired_games['Away Team'].map(team_quality)

# Extract home and away goals
desired_games[['Home Goals', 'Away Goals']] = desired_games['Result'].str.split(' - ', expand=True).astype(int)

# Add home/away indicator
desired_games['Home/Away'] = np.where(desired_games['Home Team'] == 'Man Utd', 1, 0)  # 1 for home, 0 for away

# Define sequence length
sequence_length = 2

# Function to create sequences for a single team
def create_sequences_for_team(data, team, sequence_length):
    team_games = data[(data['Home Team'] == team) | (data['Away Team'] == team)].copy()
    sequences = []
    targets = []

    for i in range(len(team_games) - sequence_length):
        # Extract the sequence of the last `sequence_length` games
        sequence = team_games.iloc[i:i+sequence_length][['Home/Away', 'Home Goals', 'Away Goals', 'Home Quality', 'Away Quality']].values
        
        # Extract the target (next game's score)
        target = team_games.iloc[i + sequence_length][['Home Goals', 'Away Goals']].values
        
        sequences.append(sequence)
        targets.append(target)

    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

# Create sequences and targets for a specific team
desired_team = 'Man Utd'
sequences, targets = create_sequences_for_team(desired_games, desired_team, sequence_length)

# Standardize data
scaler = StandardScaler()
sequences = scaler.fit_transform(sequences.reshape(-1, sequences.shape[-1])).reshape(sequences.shape)

# Convert to PyTorch tensors
sequences = torch.tensor(sequences, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.float32)

print(sequences.dtype)
print(targets.dtype)

# Define the RNN model
class ScorePredictorRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(ScorePredictorRNN, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)  # Output: 2 nodes for home and away goals

        # Initialize weights
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        
        out, _ = self.rnn(x, (h_0, c_0))
        out = out[:, -1, :]  # Take the last output of the sequence
        out = self.fc(out)
        return out

# Define model parameters
input_size = sequences.shape[2]  # Number of features per game
hidden_size = 64
num_layers = 2

# Initialize the model
model = ScorePredictorRNN(input_size, hidden_size, num_layers)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Gradient clipping threshold
clip_value = 5.0

# Training loop
num_epochs = 10000

for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    outputs = model(sequences[:-1])
    loss = criterion(outputs, targets[:-1])
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
    
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

#print(sequences)
# Example forward pass
model.eval()
with torch.no_grad():
    example_output = model(sequences[:-1])

test_output = model(sequences[-1].unsqueeze(dim=0))

file_path = 'predictions_vs_targets.csv'

with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Predicted Home Goals', 'Predicted Away Goals', 'Actual Home Goals', 'Actual Away Goals'])
    for pred, target in zip(example_output, targets):
        writer.writerow([round(pred[0].item(),3), round(pred[1].item(),3), round(target[0].item(),3), round(target[1].item(),3)])
    writer.writerow(['-','-','-','-'])
    writer.writerow(['-','-','-','-'])
    writer.writerow(['-','-','-','-'])
    writer.writerow([round(test_output[0][0].item(),3), round(test_output[0][1].item(),3), round(targets[-1][0].item(),3), round(targets[-1][0].item(),3)])

print(f"Predictions and targets have been saved to {file_path}")
