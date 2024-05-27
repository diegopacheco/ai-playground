import pandas as pd
from faker import Faker

# Initialize a Faker object
fake = Faker()

# Number of songs
num_songs = 50

# Generate song metadata
data = {
    'song_id': range(1, num_songs+1),
    'title': [fake.sentence(nb_words=3) for _ in range(num_songs)],
    'artist': [fake.name() for _ in range(num_songs)],
    'lyrics': [fake.paragraph() for _ in range(num_songs)]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('song_metadata.csv', index=False)