import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from energy_efficiency.logger import logging
from energy_efficiency.exception import EnergyEfficiencyException


class DataIngestion:
	def __init__(self, mongo_client, database_name, collection_name):
		try:
			self.mongo_client = mongo_client
			self.database_name = database_name
			self.collection_name = collection_name
		except Exception as e:
			raise EnergyEfficiencyException(e, sys)

	def initiate_data_ingestion(self):
		try:
			logging.info(f'exporting data as dataframe from mongodb database')
			df = pd.DataFrame(list(self.mongo_client[self.database_name][self.collection_name].find()))

			if '_id' in df.columns:
				df = df.drop('_id', axis=1)

			logging.info(f'creating a dataset directory')
			dataset_dir = os.path.join(os.getcwd(), 'Artifact/dataset')
			os.makedirs(dataset_dir, exist_ok=True)

			logging.info(f'spliting the dataset into training and testing')
			train_data, test_data = train_test_split(df, test_size=0.15, random_state=1)

			logging.info(f'saving train dataset in dataset directory')
			train_data.to_csv(path_or_buf=f'{dataset_dir}/train.csv', index=False, header=True)

			logging.info(f'saving test dataset in dataset directory')
			test_data.to_csv(path_or_buf=f'{dataset_dir}/test.csv', index=False, header=True)

			return train_data, test_data

		except Exception as e:
			raise EnergyEfficiencyException(e, sys)


