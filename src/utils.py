import os
import yaml
import pandas as pd

class Core_Operations:

    def load_config(self):
        config_file_path = "./config.yaml"
        config = None
        try:
            with open(config_file_path,'r') as file:
                config = yaml.safe_load(file)
                print("loading config file")
        except Exception as e:
            print("An error occured while loading the file {e}")
        return config

    def save_to_csv_file(self, df: pd.DataFrame):
        config = self.load_config()
        try:
            df.to_csv(config["data"]["csv_processed_file"],index=False)
        except Exception as e:
            print("CSv file is not generated: {e}")