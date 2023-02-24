import os
import sys
import pickle
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from energy_efficiency.logger import logging
from energy_efficiency.exception import EnergyEfficiencyException

class ModelTrainer:
	def __init__(self, train_df, test_df):
		try:
			self.train_df = train_df
			self.test_df = test_df
		except Exception as e:
			raise EnergyEfficiencyException(e, sys)

	def linear_regression(self, X, y):
		try:
			model = LinearRegression()
			model.fit(X, y)
			return model
		except Exception as e:
			raise EnergyEfficiencyException(e, sys)

	def support_vector_machine(self, X, y):
		try:
			model = SVR()
			model.fit(X, y)
			return model
		except Exception as e:
			raise EnergyEfficiencyException(e, sys)

	def random_forest(self, X, y):
		try:
			model = RandomForestRegressor()
			model.fit(X, y)
			return model
		except Exception as e:
			raise EnergyEfficiencyException(e, sys)

	def initiate_model_trainer(self):
		try:

			regression_models = {'random forest': self.random_forest,
								 'linear regression': self.linear_regression,
								 'support vector machine': self.support_vector_machine}

			logging.info(f'creating accuracy table to store the accuracy of machine learning models')
			accuracy_table = pd.DataFrame(columns=['Heating Load', 'Cooling Load'], 	
										  index=regression_models.keys())

			logging.info(f'separating input feature and output labels of train data')
			x_train = self.train_df.iloc[:, :-2]
			heat_load_train, cool_load_train = self.train_df.iloc[:, -2], self.train_df.iloc[:, -1]

			logging.info(f'separating input feature and output labels of test data')
			x_test = self.test_df.iloc[:, :-2]
			heat_load_test, cool_load_test = self.test_df.iloc[:, -2], self.test_df.iloc[:, -1]


			for i, model in enumerate(regression_models.values()):

				logging.info(f'training heating load model using {accuracy_table.index[i]}')
				heating_load_model = model(x_train, heat_load_train)

				logging.info(f'training cooling load model using {accuracy_table.index[i]}')
				cooling_load_model = model(x_train, cool_load_train)

				heating_load_predictions = heating_load_model.predict(x_test)
				heating_load_score = r2_score(heating_load_predictions, heat_load_test)
				heating_load_score = round(heating_load_score*100, 2)
				logging.info(f'heating load accuracy using {accuracy_table.index[i]}: {heating_load_score}')

				cooling_load_predictions = cooling_load_model.predict(x_test)
				cooling_load_score = r2_score(cooling_load_predictions, cool_load_test)
				cooling_load_score = round(cooling_load_score*100, 2)
				logging.info(f'cooling load accuracy using {accuracy_table.index[i]}: {cooling_load_score}')

				logging.info(f'storing the model accuracy in accuracy table')
				accuracy_table['Heating Load'][i] = heating_load_score
				accuracy_table['Cooling Load'][i] = cooling_load_score

				logging.info(f'creating a directory for saving the models')
				model_dir = f'saved models/{accuracy_table.index[i]}'
				os.makedirs(model_dir, exist_ok=True)

				logging.info(f'saving {accuracy_table.index[i]} model')
				pickle.dump(heating_load_model, open(f'{model_dir}/heating_load_model.pkl', 'wb'))
				pickle.dump(cooling_load_model, open(f'{model_dir}/cooling_load_model.pkl', 'wb'))

			accuracy_table.to_csv('model_accuracy.csv')

		except Exception as e:
			raise EnergyEfficiencyException(e, sys)