import numpy as np
import pandas as pd
import os

def generate_linear_data(version, num_samples=1000):
    """
    Generates the first and second datasets (linear relationship, different x distributions).
    All datasets will now have both x1 and x2 features.
    """
    # Set the mean for x1 based on the version
    if version == 'v1':
        mean = 10
        dataset_name = "original_data"
    else: # version == 'v2'
        mean = 20
        dataset_name = "data_drifting"

    # Generate data
    np.random.seed(42)
    x1 = np.random.normal(loc=mean, scale=2, size=num_samples)
    x2 = np.random.normal(loc=5, scale=1, size=num_samples) # x2 is consistent across datasets
    noise = np.random.normal(loc=0, scale=1, size=num_samples)
    y = 5 * x1 + 10 * x2 + 2 + noise

    df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    df['dataset_version'] = dataset_name
    return df

def generate_concept_drift_data(num_samples=1000):
    """
    Generates the third dataset (non-linear relationship).
    The relationship now includes x1^2 and x2^2 to simulate a non-linear drift.
    """
    # Generate data
    np.random.seed(42)
    x1 = np.random.normal(loc=10, scale=2, size=num_samples)
    x2 = np.random.normal(loc=5, scale=1, size=num_samples)
    noise = np.random.normal(loc=0, scale=1, size=num_samples)
    # The new non-linear relationship
    y = 5 * x1**2 + 10 * x2**2 + 2 + noise

    df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    df['dataset_version'] = 'concept_drifting'
    return df

def save_data(df, filename):
    """
    Saves the DataFrame as a CSV file to the data directory.
    """
    # Adjusted path to go to the project root's 'data' directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Data saved to: {filepath}")

if __name__ == '__main__':
    # Generate and save the original dataset
    df_original = generate_linear_data('v1')
    save_data(df_original, 'original_data.csv')

    # Generate and save the data drifting dataset
    df_data_drifting = generate_linear_data('v2')
    save_data(df_data_drifting, 'data_drifting.csv')
    
    # Generate and save the concept drifting dataset
    df_concept_drifting = generate_concept_drift_data()
    save_data(df_concept_drifting, 'concept_drifting.csv')

    print("\nAll simulated datasets have been successfully generated and saved to the data/ directory.")
