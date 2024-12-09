from batch import prepare_data
import pandas as pd
from datetime import datetime
import os
import sys

# Add the path of 'other_folder' to sys.path
sys.path.append(os.path.abspath(
    'C:/Users/Administrator/Downloads/best-practices'))


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

# Test case


def test_prepare_data():
    # Input data
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]
    columns = ['PULocationID', 'DOLocationID',
               'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    # Expected output
    expected_data = [
        (1, 1, dt(1, 2), dt(1, 10), 8.0)  # Only this row meets all conditions
    ]
    expected_columns = columns + ['duration']
    expected_df = pd.DataFrame(expected_data, columns=expected_columns)

    # Call the function
    actual_df = prepare_data(df, ['PULocationID', 'DOLocationID'])

    # Assertion
    pd.testing.assert_frame_equal(
        actual_df.reset_index(drop=True),
        expected_df.reset_index(drop=True)
    )


# Run the test
test_prepare_data()
