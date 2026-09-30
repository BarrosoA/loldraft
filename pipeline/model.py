import pandas as pd   
from sklearn.model_selection import train_test_split


df = pd.read_csv("data/training_dataset.csv")

print(df.head(5))


features = ["role","baseline_wr","lane_delta","duo_synergy","team_synergy","threat_delta"]

X = df[features]
Y = df["win"]

# 80 - 20 split

