import pandas as pd
import os
import boto3

def create_dataframe():
    """Create a DataFrame for testing purposes."""
    # Create a simple DataFrame with sample data
    data = {
        'ride_id': ['1', '2', '3'],
        'PULocationID': [1, 2, 3],
        'DOLocationID': [4, 5, 6],
        'tpep_pickup_datetime': ['2023-01-01 08:00:00', '2023-01-01 09:00:00', '2023-01-01 10:00:00'],
        'tpep_dropoff_datetime': ['2023-01-01 08:30:00', '2023-01-01 09:30:00', '2023-01-01 10:30:00'],
    }
    df = pd.DataFrame(data)
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    return df

def save_to_s3(df, input_file):
    """Save the DataFrame to S3 using Localstack."""
    # Check if the S3 endpoint URL is set for Localstack
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL', None)
    
    # Configure storage options for Localstack S3
    storage_options = {}
    if s3_endpoint_url:
        storage_options = {
            'client_kwargs': {
                'endpoint_url': s3_endpoint_url
            }
        }
    
    # Save the DataFrame to S3 (parquet format)
    df.to_parquet(input_file, engine='pyarrow', compression=None, index=False, storage_options=storage_options)
    print(f"Data successfully saved to {input_file}")

def main():
    """Main function to create and save data."""
    # Define the input file for January 2023
    input_file = 's3://nyc-duration/in/2023-01.parquet'

    # Create the DataFrame
    df_input = create_dataframe()

    # Save the DataFrame to Localstack's S3
    save_to_s3(df_input, input_file)

if __name__ == "__main__":
    main()
