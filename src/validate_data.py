import great_expectations as ge
from typing import Tuple, List
import pandas as pd
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.core.validation_definition import ValidationDefinition
from great_expectations.expectations.core import (
    expect_column_to_exist, expect_column_values_to_not_be_null, expect_column_values_to_be_in_set, expect_column_values_to_be_between,
    expect_column_pair_values_a_to_be_greater_than_b
)


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.
    
    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.
    
    """
    NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # print("🔍 Starting data validation with Great Expectations...")
    
    # Convert pandas DataFrame to Great Expectations Dataset
    context = ge.get_context()

    # Retrieve the Data Source
    data_source_name = "db_telco_churn_data_source"
    # Define the Data Asset name
    data_asset_name = "telco_data_asset"
    batch_definition_name = "db_telco_batch_name"
    data_source = context.data_sources.add_pandas(name=data_source_name)
    data_asset = data_source.add_dataframe_asset(name=data_asset_name)
    batch_definition = data_asset.add_batch_definition_whole_dataframe(name=batch_definition_name)
    suite = ExpectationSuite("validation_suite")
    # print(f"Generating Expectations for columns: {list(df.columns)}")
    imp_columns = ["customerID","gender","Partner","Dependents","PhoneService","Contract","InternetService","tenure","MonthlyCharges","TotalCharges"]
    for col in imp_columns:
        suite.add_expectation(expect_column_to_exist.ExpectColumnToExist(column=col))
    
    suite.add_expectation(expect_column_values_to_not_be_null.ExpectColumnValuesToNotBeNull(column="customerID"))

    suite.add_expectation(expect_column_values_to_be_in_set.ExpectColumnValuesToBeInSet(column="gender", value_set=["Male","Female"]))

    for col in ["Partner","Dependents","PhoneService"]:
        suite.add_expectation(expect_column_values_to_be_in_set.ExpectColumnValuesToBeInSet(column = col, value_set=["Yes","No"]))
    
    suite.add_expectation(expect_column_values_to_be_in_set.ExpectColumnValuesToBeInSet(column = "Contract", value_set=["Month-to-month", "One year", "Two year"]))
    suite.add_expectation(expect_column_values_to_be_in_set.ExpectColumnValuesToBeInSet(column = "InternetService", value_set=["DSL", "Fiber optic", "No"]))
    
    for col in ["tenure","MonthlyCharges"]:
        suite.add_expectation(expect_column_values_to_not_be_null.ExpectColumnValuesToNotBeNull(column=col))
    
    suite.add_expectation(expect_column_values_to_be_between.ExpectColumnValuesToBeBetween(column="tenure",min_value=0,max_value=240)) # 20 years
    suite.add_expectation(expect_column_values_to_be_between.ExpectColumnValuesToBeBetween(column="MonthlyCharges",min_value=0,max_value=100000))
    suite.add_expectation(expect_column_values_to_be_between.ExpectColumnValuesToBeBetween(column="TotalCharges",min_value=0,max_value=200000))

    suite.add_expectation(expect_column_pair_values_a_to_be_greater_than_b.ExpectColumnPairValuesAToBeGreaterThanB(column_A = "TotalCharges",column_B = "MonthlyCharges",or_equal = True, mostly=0.90))

    context.suites.add(suite)

    validation_df = context.validation_definitions.add(ValidationDefinition(name = "telco_validation_def",data=batch_definition,suite=suite))

    print("Run the vaidation suite")
    results = validation_df.run(batch_parameters={"dataframe":df})

    failied_expectation = []

    for result in results.results:
        if not result.success:
            print(f"Result : \n {result}")
            exp_type = result.expectation_config.type
            failied_expectation.append(exp_type)
    
    total_check = len(results.results)
    passed_checked = total_check - len(failied_expectation)

    if results.success:
        print(f"Data validation is passed {passed_checked} out of {total_check}")
    else:
        print(f"Data validation is failed {len(failied_expectation)} out of {total_check}")
        print(f"failed cases: {failied_expectation}")

    return results.success, failied_expectation 
         