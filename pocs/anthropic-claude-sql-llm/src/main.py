import anthropic
import os

client = anthropic.Anthropic(
    api_key=str(os.environ.get("ANTHROPIC_API_KEY")),
)
message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    temperature=0,
    system="Transform the following natural language requests into valid SQL queries. Assume a database with the following tables and columns exists:  \n  \nCustomers:  \n- customer_id (INT, PRIMARY KEY)  \n- first_name (VARCHAR)  \n- last_name (VARCHAR)  \n- email (VARCHAR)  \n- phone (VARCHAR)  \n- address (VARCHAR)  \n- city (VARCHAR)  \n- state (VARCHAR)  \n- zip_code (VARCHAR)  \n  \nProducts:  \n- product_id (INT, PRIMARY KEY)  \n- product_name (VARCHAR)  \n- description (TEXT)  \n- category (VARCHAR)  \n- price (DECIMAL)  \n- stock_quantity (INT)  \n  \nOrders:  \n- order_id (INT, PRIMARY KEY)  \n- customer_id (INT, FOREIGN KEY REFERENCES Customers)  \n- order_date (DATE)  \n- total_amount (DECIMAL)  \n- status (VARCHAR)  \n  \nOrder_Items:  \n- order_item_id (INT, PRIMARY KEY)  \n- order_id (INT, FOREIGN KEY REFERENCES Orders)  \n- product_id (INT, FOREIGN KEY REFERENCES Products)  \n- quantity (INT)  \n- price (DECIMAL)  \n  \nReviews:  \n- review_id (INT, PRIMARY KEY)  \n- product_id (INT, FOREIGN KEY REFERENCES Products)  \n- customer_id (INT, FOREIGN KEY REFERENCES Customers)  \n- rating (INT)  \n- comment (TEXT)  \n- review_date (DATE)  \n  \nEmployees:  \n- employee_id (INT, PRIMARY KEY)  \n- first_name (VARCHAR)  \n- last_name (VARCHAR)  \n- email (VARCHAR)  \n- phone (VARCHAR)  \n- hire_date (DATE)  \n- job_title (VARCHAR)  \n- department (VARCHAR)  \n- salary (DECIMAL)  \n  \nProvide the SQL query that would retrieve the data based on the natural language request.",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Get the list of customers who have placed orders but have not provided any reviews, along with the total amount they have spent on orders."
                }
            ]
        }
    ]
)
print(message.content)