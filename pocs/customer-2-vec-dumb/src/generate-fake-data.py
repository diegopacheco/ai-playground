import csv
from faker import Faker

# Initialize faker
fake = Faker()

# Define the number of events
num_events = 10000

# Define the product names
product_names = ['iPhone 15', 'MacBook Pro', 'AirPods Pro', 'Apple Watch', 'iPad Pro']

with open('purchase_history.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["customer_id", "ProductName"])

    for _ in range(num_events):
        writer.writerow([fake.random_int(min=1000, max=9999), fake.random_element(elements=product_names)])

print(f"Generated {num_events} fake purchase events in purchase_history.csv")