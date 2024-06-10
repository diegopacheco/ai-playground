from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the movielens-100k dataset
data = Dataset.load_builtin('ml-100k')

# Split the dataset into training and test set
trainset, testset = train_test_split(data, test_size=.25)

# Use the SVD algorithm for predictions
algo = SVD()

# Train the algorithm on the trainset
algo.fit(trainset)

# Predict ratings for the testset
predictions = algo.test(testset)

# Compute and print Root Mean Squared Error
rmse = accuracy.rmse(predictions)
print(f"Root Mean Squared Error: {rmse}")

def predict_ratings(user_id, item_ids):
    predicted_ratings = [algo.predict(user_id, item_id).est for item_id in item_ids]
    return predicted_ratings

def plot_results(item_ids, predicted_ratings):
    plt.figure(figsize=(10, 4))
    plt.bar(item_ids, predicted_ratings)
    plt.xlabel('Item ID')
    plt.ylabel('Predicted Rating')
    plt.show()

# Replace these with actual user ID and item IDs
user_id = 'USER_ID'
item_ids = ['ITEM_ID1', 'ITEM_ID2', 'ITEM_ID3']

predicted_ratings = predict_ratings(user_id, item_ids)
plot_results(item_ids, predicted_ratings)