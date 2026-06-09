CREATE EXTENSION IF NOT EXISTS plpython3u;

CREATE OR REPLACE FUNCTION llm_classify(review text)
RETURNS text
AS $$
import os, json, urllib.request

key = os.environ.get("OPENAI_API_KEY")
if not key:
    return "NO_API_KEY"

payload = json.dumps({
    "model": "gpt-4o-mini",
    "temperature": 0,
    "messages": [
        {"role": "system", "content": "You are a sentiment classifier. Reply with exactly one word: POSITIVE, NEGATIVE or NEUTRAL."},
        {"role": "user", "content": review}
    ]
}).encode("utf-8")

req = urllib.request.Request(
    "https://api.openai.com/v1/chat/completions",
    data=payload,
    headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}
)
resp = urllib.request.urlopen(req, timeout=30)
data = json.loads(resp.read().decode("utf-8"))
return data["choices"][0]["message"]["content"].strip()
$$ LANGUAGE plpython3u;

CREATE TABLE reviews (
    id     serial PRIMARY KEY,
    review text NOT NULL
);

INSERT INTO reviews (review) VALUES
('The product exceeded my expectations, absolutely love it!'),
('Terrible experience, the item broke after a single day of use.'),
('It works as described, nothing special but it does the job.'),
('Best purchase I have made all year, I highly recommend it.'),
('Complete waste of money, I would never buy this again.');
