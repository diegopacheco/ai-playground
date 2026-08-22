AGENTS = ["billing", "tech_support", "sales", "abuse", "general"]
SENTIMENTS = ["positive", "neutral", "negative"]

AGENT_HINTS = {
    "billing": "invoices, charges, refunds, payment methods, pricing on an existing account",
    "tech_support": "errors, outages, bugs, something not working",
    "sales": "buying, upgrading, plans, quotes, trials, contacting sales",
    "abuse": "spam, scams, phishing, unsolicited advertising",
    "general": "anything else, greetings, small talk, unclear requests",
}

REQUESTS = [
    ("I was charged twice for the same invoice this month.", "billing", "negative"),
    ("Can you refund the annual plan? I cancelled last week.", "billing", "negative"),
    ("Where do I update the credit card on my account?", "billing", "neutral"),
    ("The receipt for order 8823 never arrived by email.", "billing", "neutral"),
    ("Thanks for fixing the double charge so fast!", "billing", "positive"),
    ("My invoice shows a currency I did not select.", "billing", "negative"),
    ("How is the annual discount applied to my next bill?", "billing", "neutral"),
    ("The API returns 502 on every request since this morning.", "tech_support", "negative"),
    ("Login fails with 'invalid token' after the update.", "tech_support", "negative"),
    ("Uploads hang at 90 percent and never finish.", "tech_support", "negative"),
    ("How do I rotate the API key without downtime?", "tech_support", "neutral"),
    ("The dashboard is much faster after the last release, nice work.", "tech_support", "positive"),
    ("Webhooks stopped firing and I see no errors in the log.", "tech_support", "negative"),
    ("Is there a way to export my data as CSV?", "tech_support", "neutral"),
    ("What is the price difference between Pro and Enterprise?", "sales", "neutral"),
    ("We need a quote for 200 seats.", "sales", "neutral"),
    ("Can I extend my trial by two weeks?", "sales", "neutral"),
    ("I want to upgrade my team to the yearly plan.", "sales", "positive"),
    ("Does the Enterprise tier include SSO?", "sales", "neutral"),
    ("Please have someone from sales call me tomorrow.", "sales", "neutral"),
    ("CONGRATULATIONS you won a free iPhone, click here now!!!", "abuse", "negative"),
    ("Buy cheap followers, 10000 for $5, limited offer.", "abuse", "negative"),
    ("Your account will be closed unless you verify your password here.", "abuse", "negative"),
    ("Earn $5000 a week working from home, no experience needed.", "abuse", "negative"),
    ("URGENT: wire transfer needed, reply with your bank details.", "abuse", "negative"),
    ("Hello, is anyone there?", "general", "neutral"),
    ("Just wanted to say the team has been great to work with.", "general", "positive"),
    ("What are your office hours?", "general", "neutral"),
    ("I have a question but I am not sure who to ask.", "general", "neutral"),
    ("Good morning from Lisbon.", "general", "positive"),
]
