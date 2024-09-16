import numpy as np
import random
import os
import json

class Team:
    def __init__(self, name):
        self.name = name
        self.matches_played = 0
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.goals_scored = 0
        self.goals_conceded = 0
        self.goal_difference = 0
        self.points = 0

    def update_stats(self, goals_scored, goals_conceded):
        self.goals_scored += goals_scored
        self.goals_conceded += goals_conceded
        self.goal_difference += (goals_scored - goals_conceded)
        self.matches_played += 1
        
        # Update results based on score
        if goals_scored > goals_conceded:
            self.wins += 1
            self.points += 3
        elif goals_scored == goals_conceded:
            self.draws += 1
            self.points += 1
        else:
            self.losses += 1

    def to_dict(self):
        return {
            'name': self.name,
            'matches_played': self.matches_played,
            'wins': self.wins,
            'draws': self.draws,
            'losses': self.losses,
            'goals_scored': self.goals_scored,
            'goals_conceded': self.goals_conceded,
            'goal_difference': self.goal_difference,
            'points': self.points
        }

    @classmethod
    def from_dict(cls, data):
        team = cls(data['name'])
        team.matches_played = data['matches_played']
        team.wins = data['wins']
        team.draws = data['draws']
        team.losses = data['losses']
        team.goals_scored = data['goals_scored']
        team.goals_conceded = data['goals_conceded']
        team.goal_difference = data['goal_difference']
        team.points = data['points']
        return team

    def __str__(self):
        return f"{self.name}: {self.points} points, {self.goal_difference} GD, {self.matches_played} games played"

class League:
    def __init__(self, teams):
        self.initial_teams = teams
        self.teams = {team.name: team for team in teams}
        self.gameweek_file = 'current_gameweek.json'
        self.teams_file = 'teams.json'

    def load_gameweek(self):
        if os.path.exists(self.gameweek_file):
            with open(self.gameweek_file, 'r') as f:
                data = json.load(f)
                return data.get('gameweek', 0)
        return 0

    def save_gameweek(self, gameweek):
        with open(self.gameweek_file, 'w') as f:
            json.dump({'gameweek': gameweek}, f)

    def load_teams(self):
        if os.path.exists(self.teams_file):
            with open(self.teams_file, 'r') as f:
                data = json.load(f)
                self.teams = {team_data['name']: Team.from_dict(team_data) for team_data in data}
        else:
            self.teams = {team.name: team for team in self.initial_teams}

    def save_teams(self):
        with open(self.teams_file, 'w') as f:
            json.dump([team.to_dict() for team in self.teams.values()], f)

    def reset_league(self):
        self.teams = {team.name: Team(team.name) for team in self.initial_teams}
        self.save_teams()  # Save initial state

        # Reset gameweek file
        with open(self.gameweek_file, 'w') as f:
            json.dump({'gameweek': 0}, f)  # Start from gameweek 0

    def play_match(self, home_team_name, away_team_name, home_goals, away_goals):
        home_team = self.teams[home_team_name]
        away_team = self.teams[away_team_name]

        home_team.update_stats(home_goals, away_goals)
        away_team.update_stats(away_goals, home_goals)
        
    def display_table(self):
        gameweek = self.load_gameweek()
        filename = 'league_table.txt'
        
        # Extract points, goal difference, and matches played into arrays
        points = np.array([team.points for team in self.teams.values()])
        goal_difference = np.array([team.goal_difference for team in self.teams.values()])
        games_played = np.array([team.matches_played for team in self.teams.values()])
        
        # Sort by points (primary) and goal difference (secondary)
        sorted_indices = np.lexsort((-goal_difference, -points))  # Negative for descending order
        
        sorted_teams = [list(self.teams.values())[i] for i in sorted_indices]

        with open(filename, 'w') as f:  # Open file in append mode
            f.write(f"\nGameweek {gameweek +1}\n")
            f.write("Team, Points, GD, Games Played\n")
            for team in sorted_teams:
                f.write(f"{team.name}, {team.points}, {team.goal_difference}, {team.matches_played}\n")
                
        print(f"League table for gameweek {gameweek +1} has been written to '{filename}'.")

    def play_gameweek(self):
        # Shuffle teams to randomize matchups
        random.shuffle(list(self.teams.values()))
        
        # Loop through the teams in pairs
        for i in range(0, len(self.teams), 2):
            home_team = list(self.teams.values())[i]
            away_team = list(self.teams.values())[i + 1]
            
            # Randomize goals for both teams for simplicity
            home_goals = random.randint(0, 5)
            away_goals = random.randint(0, 5)
            
            # Play the match
            self.play_match(home_team.name, away_team.name, home_goals, away_goals)
            print(f"Match Result: {home_team.name} {home_goals} - {away_goals} {away_team.name}")

    def simulate_season(self):
        self.load_teams()  # Load saved team states
        current_gameweek = self.load_gameweek()
        new_gameweek = current_gameweek + 1
        print(f"\nGameweek {new_gameweek}")
        
        self.play_gameweek()
        self.display_table()
        self.save_gameweek(new_gameweek)
        self.save_teams()  # Save team states

# List of team names
team_names = [
    'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 'Chelsea', 'Crystal Palace', 
    'Everton', 'Fulham', 'Ipswich', 'Leicester', 'Liverpool', 'Man City', 'Man UTD', 'Newcastle', 
    'Notts Forest', 'Southampton', 'Tottenham', 'West Ham', 'Wolves'
]

# Instantiate Team objects for each team
teams = [Team(name) for name in team_names]

# Instantiate the league with the list of Team objects
premier_league = League(teams)

# Simulate the season
premier_league.simulate_season()

# To reset the league:
# premier_league.reset_league()
# print("League has been reset.")
