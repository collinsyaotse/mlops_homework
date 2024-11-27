import argparse
import pickle
import pandas as pd

# Load the pre-trained model and DictVectorizer
with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Define categorical features
categorical = ['PULocationID', 'DOLocationID']


# Function to read and preprocess data
def read_data(filename):
    df = pd.read_parquet(filename)

    # Calculate trip duration in minutes
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df['duration'].dt.total_seconds() / 60

    # Filter trips based on reasonable duration
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)].copy()

    # Handle categorical columns
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


# Main script logic
def main(year, month):
    # Construct the file URL dynamically
    filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'

    # Load and preprocess the data
    df = read_data(filename)

    # Transform the data for prediction
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)

    # Make predictions
    y_pred = model.predict(X_val)

    # Add ride_id for identification
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    # Create a DataFrame with the results
    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'prediction': y_pred
    })

    # Define the output file path
    output_file = 'results.parquet'

    # Save the DataFrame to a Parquet file
    df_result.to_parquet(
        output_file,
        engine='pyarrow',  # Ensure pyarrow is installed
        compression=None,
        index=False
    )

    # Calculate and print the mean predicted duration
    mean_predicted_duration = y_pred.mean()
    print(f"Mean predicted duration: {mean_predicted_duration:.2f}")

    print(f"Results saved to {output_file}")


# Ensure the script runs only when executed directly
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process NYC taxi trip data")
    parser.add_argument("--year", type=int, required=True,
                        help="Year for the data")
    parser.add_argument("--month", type=int, required=True,
                        help="Month for the data")

    args = parser.parse_args()
    main(args.year, args.month)
