import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from energy_efficiency.logger import logging
from energy_efficiency.exception import EnergyEfficiencyException

class ModelTrainer:
	def __init__(self, train_df, test_df):
		try:
			self.train_df = train_df
			self.test_df = test_df
		except Exception as e:
			raise EnergyEfficiencyException(e, sys)


	def model_evaluation(self, model, x_test, y_test):
		try:
			prediction = model.predict(x_test)
			r2 = round(r2_score(prediction, y_test), 3)
			mse = round(mean_squared_error(prediction, y_test), 3)
			rmse = round(np.sqrt(mse), 3)
			mae = round(mean_absolute_error(prediction, y_test), 3)
			return r2, mse, rmse, mae

		except Exception as e:
			raise EnergyEfficiencyException(e, sys)


	def initiate_model_trainer(self):
		try:

			regression_models = {'random forest': RandomForestRegressor(),
								 'linear regression': LinearRegression(),
								 'support vector machine': SVR(),
								 'decision tree': DecisionTreeRegressor()}

			logging.info(f'creating accuracy table to store the accuracy of machine learning models')
			heating_metrics = pd.DataFrame(columns=['Accuracy', 'MSE', 'RMSE', 'MAE'], dtype=object,
										   index=regression_models.keys())

			cooling_metrics = pd.DataFrame(columns=['Accuracy', 'MSE', 'RMSE', 'MAE'], dtype=object,
										   index=regression_models.keys())

			logging.info(f'separating input feature and output labels of train data')
			x_train = self.train_df.iloc[:, :-2]
			heat_load_train, cool_load_train = self.train_df.iloc[:, -2], self.train_df.iloc[:, -1]

			logging.info(f'separating input feature and output labels of test data')
			x_test = self.test_df.iloc[:, :-2]
			heat_load_test, cool_load_test = self.test_df.iloc[:, -2], self.test_df.iloc[:, -1]


			for i, model in enumerate(regression_models.values()):

				logging.info(f'training heating load model using {heating_metrics.index[i]}')
				heating_load_model = model.fit(x_train, heat_load_train)

				logging.info(f'training cooling load model using {cooling_metrics.index[i]}')
				cooling_load_model = model.fit(x_train, cool_load_train)

				r2_score, mse, rmse, mae = self.model_evaluation(model=model, x_test=x_test, y_test=heat_load_test)
				heating_metrics['Accuracy'][i] = r2_score
				heating_metrics['MSE'][i] = mse
				heating_metrics['RMSE'][i] = rmse
				heating_metrics['MAE'][i] = mae
				
				r2_score, mse, rmse, mae = self.model_evaluation(model=model, x_test=x_test, y_test=cool_load_test)
				cooling_metrics['Accuracy'][i] = r2_score
				cooling_metrics['MSE'][i] = mse
				cooling_metrics['RMSE'][i] = rmse
				cooling_metrics['MAE'][i] = mae

				logging.info(f'creating a directory for saving the models')
				model_dir = f'saved models/{heating_metrics.index[i]}'
				os.makedirs(model_dir, exist_ok=True)

				logging.info(f'saving {heating_metrics.index[i]} model')
				pickle.dump(heating_load_model, open(f'{model_dir}/heating_load_model.pkl', 'wb'))
				pickle.dump(cooling_load_model, open(f'{model_dir}/cooling_load_model.pkl', 'wb'))

			heating_metrics.to_csv('heating_metrics.csv')
			cooling_metrics.to_csv('cooling_metrics.csv')

		except Exception as e:
			raise EnergyEfficiencyException(e, sys)