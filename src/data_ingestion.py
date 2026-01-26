import os
import csv
import pandas as pd
import numpy as np
from utils import Core_Operations

class DataIngestion:

    def load_processed_csv_file(self):
        config = Core_Operations().load_config()
        csv_file_path = config["data"]["csv_processed_file"]
        data = None
        try:
            data = pd.read_csv(csv_file_path)
        except Exception as e:
            print("An error while loading the csv file")
        return data
    
    def load_raw_csv_file(self):
        config = Core_Operations().load_config()
        csv_file_path = config["data"]["csv_raw_file"]
        data = None
        try:
            data = pd.read_csv(csv_file_path)
        except Exception as e:
            print("An error while loading the csv file")
        return data
