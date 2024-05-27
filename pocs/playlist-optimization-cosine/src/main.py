import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# columns: ['user_id', 'song_id']
user_song_df = pd.read_csv('user_song_interactions.csv')  

# columns: ['song_id', 'title', 'artist', 'lyrics']
song_meta_df = pd.read_csv('song_metadata.csv')  
print(song_meta_df)

print(f" ids on user_song_df {user_song_df['song_id'].unique()}")
print(f" ids on song_meta_df {song_meta_df['song_id'].unique()}")

print(f" unique song ids in user_song_interactions.csv =  {user_song_df['song_id'].unique().size}")
print(f" unique song ids in song_metadata = {song_meta_df['song_id'].unique().size}")

print(f" all columns in user_song_df {user_song_df.columns}")
print(f" all columns in song_meta_df {song_meta_df.columns}")

# Now you can merge
user_song_df['song_id'] = user_song_df['song_id'].astype(int)
song_meta_df['song_id'] = song_meta_df['song_id'].astype(int)

df = pd.merge(user_song_df, song_meta_df, on='song_id', how='inner')
print(f"merged dataframe = {df}")
cosine_sim = None

# Check if df is empty
if df.empty:
    print("Merged dataframe is empty. Please check your 'song_id' values.")
else:
    # Create a TF-IDF matrix of the lyrics
    tfidf = TfidfVectorizer(min_df=1, stop_words=None)
    tfidf_matrix = tfidf.fit_transform(df['lyrics'])

    # Compute the cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print(cosine_sim)

# Function that takes in a song title as input and outputs most similar songs
def get_recommendations(title, cosine_sim=cosine_sim):
    print(f"Dataset for recommendations:\n{df}")

    # Check if the song title exists in the DataFrame
    if title not in df['title'].values:
        return f"Song title '{title}' not found in the dataset."
    
    # Get the index of the song that matches the title
    idx = df[df['title'] == title].index[0]

    # Get the pairwise similarity scores of all songs with that song
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the songs based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar songs
    sim_scores = sim_scores[1:11]

    # Get the song indices
    song_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar songs
    return df['title'].iloc[song_indices]

# Get recommendations for a specific song
recommendations = get_recommendations('Sabbath bloody sabbath')
print(f" Recommended songs for 'Sabbath bloody sabbath':\n{recommendations}")

# Plot a word cloud of the recommended song titles
wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(' '.join(recommendations))
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()