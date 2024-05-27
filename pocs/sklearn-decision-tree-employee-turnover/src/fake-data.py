import pandas as pd
import numpy as np

# Set a seed for reproducibility
np.random.seed(0)

# Generate random data
n_samples = 1000
satisfaction = np.random.rand(n_samples)
tenure = np.random.randint(1, 10, size=n_samples)
performance = np.random.rand(n_samples)
left_company = np.random.randint(0, 2, size=n_samples)

# Create a DataFrame
df = pd.DataFrame({
    'satisfaction': satisfaction,
    'tenure': tenure,
    'performance': performance,
    'left_company': left_company
})

# Save the DataFrame to a CSV file
df.to_csv('employee_data.csv', index=False)