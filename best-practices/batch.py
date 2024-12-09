import sys
import pickle
import pandas as pd
import os

def prepare_data(df: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    """Transforms the data by applying necessary preprocessing steps."""
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    # Filter rows based on duration
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

def get_input_path(year, month):
    """Get the input path for reading the data."""
    # Default input pattern if the env variable is not set
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)

def get_output_path(year, month):
    """Get the output path for saving the predictions."""
    # Default output pattern if the env variable is not set
    default_output_pattern = 's3://nyc-duration/out/{year:04d}-{month:02d}.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)

def read_data(filename, categorical):
    """Reads the Parquet file and preprocesses it."""
    # Check if S3_ENDPOINT_URL is set for Localstack
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL', None)
    
    # Configure storage options to specify the endpoint URL
    storage_options = {}
    if s3_endpoint_url:
        storage_options = {
            'client_kwargs': {
                'endpoint_url': s3_endpoint_url
            }
        }
    
    # Read the parquet file from S3 or local depending on the endpoint
    df = pd.read_parquet(filename, storage_options=storage_options)
    
    # Preprocess the data
    return prepare_data(df, categorical)

def main(year, month):
    """Main function to process data and make predictions for a given year and month."""
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    # Load the model
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']
    
    # Read and preprocess the data
    df = read_data(input_file, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    # Prepare features and make predictions
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    # Print the predicted mean duration
    print('predicted mean duration:', y_pred.mean())

    # Save the results
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(output_file, engine='pyarrow', index=False)

if __name__ == "__main__":
    # Parse command-line arguments
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    # Run the main function
    main(year, month)
