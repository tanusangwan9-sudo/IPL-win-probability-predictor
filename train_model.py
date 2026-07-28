import pandas as pd
import numpy as np
import pickle
matches = pd.read_csv("matches.csv")

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load datasets
matches = pd.read_csv("matches.csv")
deliveries = pd.read_csv("deliveries.csv")

# First innings score
total_score_df = deliveries.groupby(['match_id', 'inning']).sum()['total_runs'].reset_index()
total_score_df = total_score_df[total_score_df['inning'] == 1]

# Merge
match_df = matches.merge(total_score_df[['match_id', 'total_runs']],
                         left_on='id',
                         right_on='match_id')

# Keep required matches
teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Daredevils'
]

match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]

match_df = match_df[['match_id', 'city', 'winner', 'total_runs']]

# Second innings
delivery_df = match_df.merge(deliveries, on='match_id')

delivery_df = delivery_df[delivery_df['inning'] == 2]

delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()

delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']

delivery_df['balls_left'] = 126 - (delivery_df['over'] * 6 + delivery_df['ball'])

delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x: 0 if x == "0" else 1)

delivery_df['wickets'] = delivery_df.groupby('match_id')['player_dismissed'].cumsum()
delivery_df['wickets_left'] = 10 - delivery_df['wickets']

delivery_df['crr'] = (delivery_df['current_score'] * 6) / (120 - delivery_df['balls_left'])

delivery_df['rrr'] = (delivery_df['runs_left'] * 6) / delivery_df['balls_left']

def result(row):
    return 1 if row['batting_team'] == row['winner'] else 0

delivery_df['result'] = delivery_df.apply(result, axis=1)

final_df = delivery_df[['batting_team',
                        'bowling_team',
                        'city',
                        'runs_left',
                        'balls_left',
                        'wickets_left',
                        'total_runs_x',
                        'crr',
                        'rrr',
                        'result']]

final_df = final_df.sample(final_df.shape[0])

final_df.dropna(inplace=True)
import numpy as np

final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
final_df.dropna(inplace=True)

X = final_df.iloc[:, :-1]
y = final_df.iloc[:, -1]

trf = ColumnTransformer([
    ('trf', OneHotEncoder(drop='first'),
     ['batting_team', 'bowling_team', 'city'])
], remainder='passthrough')

pipe = Pipeline([
    ('step1', trf),
    ('step2', LogisticRegression(solver='liblinear'))
])

pipe.fit(X, y)

pickle.dump(pipe, open("pipe.pkl", "wb"))

print("pipe.pkl created successfully")