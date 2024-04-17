import os
from pinecone import Pinecone

api_key = str(os.getenv("PINECONE_API_KEY"))
print(f"using API key {api_key}")

pc = Pinecone(api_key=api_key)
index = pc.Index("test-index-text-embedding-ada-002")

# Create a vector of dimension 1536
vector = [0.1] * 1536

print(index.upsert(
    vectors=[
        {
            "id": "vec1", 
            "values": vector, 
            "metadata": {"genre": "drama"}
        }, {
            "id": "vec2", 
            "values": vector, 
            "metadata": {"genre": "action"}
        }, {
            "id": "vec3", 
            "values": vector, 
            "metadata": {"genre": "drama"}
        }, {
            "id": "vec4", 
            "values": vector, 
            "metadata": {"genre": "action"}
        }
    ],
    namespace= "ns1"
))

print(index.query(
    namespace="ns1",
    vector=vector,
    top_k=2,
    include_values=True,
    include_metadata=True,
    filter={"genre": {"$eq": "action"}}
))