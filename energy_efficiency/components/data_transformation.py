import os
import sys
import pickle
import pandas as pd
from energy_efficiency.logger import logging
from energy_efficiency.exception import EnergyEfficiencyException
from sklearn.preprocessing import StandardScaler


class DataTransformation:
	def __init__(self, train_data, test_data):
		try:
			self.train_data = train_data
			self.test_data = test_data

		except Exception as e:
			raise EnergyEfficiencyException(e, sys)


	def initiate_data_transformation(self):
		try:
			train_data = self.train_data
			test_data = self.test_data

			features = train_data.columns[:-2]
			xtrain = train_data[features]
			xtest = test_data[features]

			scaler = StandardScaler()
			train_data[features] = scaler.fit_transform(xtrain)
			test_data[features] = scaler.transform(xtest)

			train_data = pd.DataFrame(train_data)
			test_data = pd.DataFrame(test_data)

			transformer_dir = os.path.join(os.getcwd(), 'Artifact/transformer')
			os.makedirs(transformer_dir, exist_ok=True)

			pickle.dump(scaler, open(f'{transformer_dir}/scaler.pkl', 'wb'))

			return train_data, test_data

		except Exception as e:
			raise EnergyEfficiencyException(e, sys)
		
