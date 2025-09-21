import numpy as np
import pandas as pd
import os
import argparse

def generate_linear_data(version, num_samples=1000):
    """
    Generates the first and second datasets (linear relationship, different x distributions).
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
    x2 = np.random.normal(loc=5, scale=1, size=num_samples)
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
    # Corrected path to navigate from the script location to the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Data saved to: {filepath}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a specific type of simulated dataset.")
    parser.add_argument('data_type', choices=['original', 'data_drifting', 'concept_drifting'],
                        help="The type of data to generate.")
    args = parser.parse_args()

    output_filename = 'current_data.csv'
    
    if args.data_type == 'original':
        df = generate_linear_data('v1')
    elif args.data_type == 'data_drifting':
        df = generate_linear_data('v2')
    elif args.data_type == 'concept_drifting':
        df = generate_concept_drift_data()

    save_data(df, output_filename)

    print(f"\nSuccessfully generated '{args.data_type}' data to {output_filename}.")