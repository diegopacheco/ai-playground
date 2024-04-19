import pandas as pd
from gensim.models import Word2Vec

# Step 1: Load and preprocess the data
df = pd.read_csv('purchase_history.csv')
purchases = df.groupby('customer_id')['ProductName'].apply(list).tolist()

# Step 2: Train the customer2vec model
model = Word2Vec(purchases, vector_size=100, window=5, min_count=1, workers=4)

# Step 3: Save the model
model.save("customer2vec.model")

# Step 4: Load the model and use it for product recommendation
class ProductRecommender:
    def __init__(self, model_path):
        # Load the trained customer2vec model
        self.model = Word2Vec.load(model_path)

    def recommend_product(self, event):
        # Use the model to find the product that is most similar to the event
        similar_products = self.model.wv.most_similar(positive=[event], topn=1)

        # If there are any similar products, return the one with the highest similarity score
        if similar_products:
            return similar_products[0][0]
        else:
            # If there are no similar products, return a default product
            return "Gift Card"

# Test the ProductRecommender class
recommender = ProductRecommender('customer2vec.model')
print("iPhone 15   - recommendation: " + recommender.recommend_product("iPhone 15"))    # Outputs: Most similar product to "iPhone 15"
print("Apple Watch - recommendation: " + recommender.recommend_product("Apple Watch"))  # Outputs: Most similar product to "Apple Watch"