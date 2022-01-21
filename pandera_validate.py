import pandera as pa
import pandas as pd
from abc_functions import abc_classification, load_data, filter_dataset
from xyz_functions import xyz_classifier

#---------- Validation 1 ----------#
"""
Validating functions:
(1) load_data(data.csv) -> loading csv data
(2) filter_dataset(parameters) -> filtering dataset and converting Date attribute into datetime datatype
"""

# Loading the data
data = load_data('Product Demand 6 Months.csv')
dataset = filter_dataset(data, 'Last 3 Months')

# Taking a subsample
data_sample = dataset.sample(n=20, random_state=999)

# defining the schema
schema_1 = pa.DataFrameSchema({
    # this column should be a string and not null
    "Product_Code": pa.Column(pa.String, nullable=False),
    # this column should be a datetime and not null
    "Date": pa.Column(pa.DateTime, nullable=False),
    # this column should be an integer and not null
    "Order_Demand": pa.Column(pa.Int,
                              nullable=False,
                              checks=pa.Check(lambda s: (s >= 0)))
})

# Validating the data
print(schema_1.validate(data_sample, lazy=True))

#---------- End of Validation 1 ----------#

print("--------------------")

#---------- Validation 2 ----------#
"""
Validating functions:
abc_classification(parameters) -> classifying products based on demand volume
"""
# making abc classification
data_abc, abc_class = abc_classification(dataset, 5, 15)

# taking sample
data_abc_sample = data_abc.sample(n=20, random_state=999)

# defining the schema
schema_2 = pa.DataFrameSchema({
    # this column should be a string and not null
    "Product_Code": pa.Column(pa.String,
                              nullable=False,
                              checks=pa.Check(lambda s: s.duplicated().sum() == 0)),  # make sure that all product code is unique from each other
    "Order_Demand": pa.Column(pa.Int,
                              nullable=False,
                              checks=pa.Check(lambda s: (s >= 0))),
    "rank": pa.Column(pa.Int, nullable=False),
    "total": pa.Column(pa.Int, nullable=False),
    "rank_cumsum": pa.Column(pa.Float, nullable=False),
    "class": pa.Column(pa.String, nullable=False)
})

# validating the data
print(schema_2(data_abc_sample, lazy=True))

#---------- End of Validation 2 ----------#

print("--------------------")

#---------- Validation 3 ----------#
"""
Validating functions: 
xyz_classifier(parameters) 
"""
# making xyz classification
data_xyz, xyz_monthly, xyz_demand = xyz_classifier(dataset, 10, 15)

# taking sample
data_xyz_sample = data_xyz.sample(n=20, random_state=999)

# defining the schema
schema_3 = pa.DataFrameSchema({
    "Product_Code": pa.Column(pa.String,
                              nullable=False,
                              checks=pa.Check(lambda s: s.duplicated().sum() == 0)),  # make sure that all product code is unique from each other
    "std": pa.Column(pa.Float, nullable=False),
    "total": pa.Column(pa.Float, nullable=False),
    "avg": pa.Column(pa.Float, nullable=False),
    "cov": pa.Column(pa.Float, nullable=False),
    "total_product": pa.Column(pa.Int, nullable=False),
    "rank_cumsum": pa.Column(pa.Float, nullable=False),
    "xyz_class": pa.Column(pa.String, nullable=False)
})

# validating the data
print(schema_3(data_xyz_sample, lazy=True))
