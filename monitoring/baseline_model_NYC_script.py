# -*- coding: utf-8 -*-

import requests
import datetime
import pandas as pd

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric
import pandas as pd
from joblib import load, dump
from tqdm import tqdm
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

from evidently.ui.workspace import Workspace
from evidently.ui.dashboards import DashboardPanelCounter, DashboardPanelPlot, CounterAgg, PanelValue, PlotType, ReportFilter
from evidently.renderers.html_widgets import WidgetSize

import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from evidently.metrics import ColumnQuantileMetric


url = "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-03.parquet"
# Download the file
response = requests.get(url)

# Save the file to disk
with open("dataset.csv", "wb") as file:
    file.write(response.content)

print("Dataset downloaded and saved as dataset.csv")

df_march = pd.read_parquet('dataset.csv')

# change jan_data to df_march
jan_data = df_march
# create target
jan_data["duration_min"] = jan_data.lpep_dropoff_datetime - \
    jan_data.lpep_pickup_datetime
jan_data.duration_min = jan_data.duration_min.apply(
    lambda td: float(td.total_seconds())/60)

# filter out outliers
jan_data = jan_data[(jan_data.duration_min >= 0) &
                    (jan_data.duration_min <= 60)]
jan_data = jan_data[(jan_data.passenger_count > 0) &
                    (jan_data.passenger_count <= 8)]

jan_data.duration_min.hist()

# data labeling
target = "duration_min"
num_features = ["passenger_count",
                "trip_distance", "fare_amount", "total_amount"]
cat_features = ["PULocationID", "DOLocationID"]

jan_data.shape

train_data = jan_data[:30000]
val_data = jan_data[30000:]

model = LinearRegression()

model.fit(train_data[num_features + cat_features], train_data[target])

train_preds = model.predict(train_data[num_features + cat_features])
train_data['prediction'] = train_preds

val_preds = model.predict(val_data[num_features + cat_features])
val_data['prediction'] = val_preds

print(mean_absolute_error(train_data.duration_min, train_data.prediction))
print(mean_absolute_error(val_data.duration_min, val_data.prediction))

"""# Dump model and reference data"""

# dump model and reference data
dump(model, 'model.joblib')

train_data.to_parquet("train_data.parquet")
val_data.to_parquet("val_data.parquet")

"""# Evidently Report"""

column_mapping = ColumnMapping(
    target=None,
    prediction='prediction',
    numerical_features=num_features,
    categorical_features=cat_features
)

report = Report(metrics=[
    ColumnDriftMetric(column_name='prediction'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric(),
    ColumnQuantileMetric(column_name='fare_amount', quantile=0.5)
]
)

report.run(reference_data=train_data, current_data=val_data,
           column_mapping=column_mapping)

report.show(mode='inline')

result = report.as_dict()

result

result['metrics'][3]

"""#EVIDENTLY MONITORING DASHBOARD"""


val_data['lpep_pickup_datetime'] = pd.to_datetime(
    val_data['lpep_pickup_datetime'])

val_data = val_data.dropna(subset=['lpep_pickup_datetime'])

ws = Workspace("workspace")

project = ws.create_project("NYC Taxi Data Quality Project")
project.description = "My project descriotion"
project.save()

regular_report = Report(
    metrics=[
        DataQualityPreset(),
        ColumnQuantileMetric(column_name='fare_amount', quantile=0.5),
    ],
    timestamp=datetime.datetime(2024, 11, 29)
)

# Specify both left and right bounds for the between method
regular_report.run(reference_data=None,
                   current_data=val_data.loc[val_data.lpep_pickup_datetime.between(
                       '2024-03-1', '2024-03-31')],  # Added right bound
                   column_mapping=column_mapping)

regular_report

regular_report = Report(
    metrics=[
        DataQualityPreset(),  # Checks for data quality issues
        ColumnQuantileMetric(column_name='fare_amount',
                             quantile=0.5),  # Median of fare_amount
    ],
    # The timestamp of when the report is being generated
    timestamp=datetime.datetime(2024, 11, 29)
)


march_data = val_data.loc[val_data.lpep_pickup_datetime.between(
    '2024-03-01', '2024-03-31')]

# Ensure data is grouped by day
# Extract date part from datetime
march_data['pickup_date'] = march_data['lpep_pickup_datetime'].dt.date


regular_report.run(
    reference_data=None,
    current_data=march_data,
    column_mapping=column_mapping
)

daily_medians = march_data.groupby('pickup_date')['fare_amount'].median()

# Now get the maximum of these daily medians
max_daily_median = daily_medians.max()

print(
    f"The maximum value of the 0.5 quantile (median) for 'fare_amount' during March 2024 is: {max_daily_median}")

ws.add_report(project.id, regular_report)

# configure the dashboard
project.dashboard.add_panel(
    DashboardPanelCounter(
        filter=ReportFilter(metadata_values={}, tag_values=[]),
        agg=CounterAgg.NONE,
        title="NYC taxi data dashboard"
    )
)

project.dashboard.add_panel(
    DashboardPanelPlot(
        filter=ReportFilter(metadata_values={}, tag_values=[]),
        title="Inference Count",
        values=[
            PanelValue(
                metric_id="DatasetSummaryMetric",
                field_path="current.number_of_rows",
                legend="count"
            ),
        ],
        plot_type=PlotType.BAR,
        size=WidgetSize.HALF,
    ),
)

project.dashboard.add_panel(
    DashboardPanelPlot(
        filter=ReportFilter(metadata_values={}, tag_values=[]),
        title="Number of Missing Values",
        values=[
            PanelValue(
                metric_id="DatasetSummaryMetric",
                field_path="current.number_of_missing_values",
                legend="count"
            ),
        ],
        plot_type=PlotType.LINE,
        size=WidgetSize.HALF,
    ),
)

project.dashboard.add_panel(
    DashboardPanelPlot(
        filter=ReportFilter(metadata_values={}, tag_values=[]),
        title="fare amount",
        values=[
            PanelValue(
                metric_id="ColumnQuantileMetric",
                field_path="fare_amount",
                legend="count"
            ),
        ],
        plot_type=PlotType.BAR,
        size=WidgetSize.HALF,
    ),
)

project.save()
