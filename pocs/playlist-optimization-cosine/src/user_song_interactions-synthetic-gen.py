import pandas as pd
import numpy as np

# Number of users and songs
num_users = 100
num_songs = 50

# Generate random user-song interactions
data = {
    'user_id': np.random.choice(range(1, num_users+1), size=1000),
    'song_id': np.random.choice(range(1, num_songs), size=1000)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('user_song_interactions.csv', index=False)