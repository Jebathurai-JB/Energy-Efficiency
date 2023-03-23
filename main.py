import os
import sys
import pymongo
from energy_efficiency.components.data_ingestion import DataIngestion
from energy_efficiency.components.model_training import ModelTrainer
from energy_efficiency.components.data_transformation import DataTransformation
from energy_efficiency.exception import EnergyEfficiencyException

database_name = "Ineuron_project"
collection_name = "Energy_Efficiency"
mongo_client = pymongo.MongoClient("mongodb+srv://jebathurai:<password>@cluster0.irlqnsu.mongodb.net/?retryWrites=true&w=majority")


if __name__ == '__main__':

	try:
		data_ingestion = DataIngestion(mongo_client=mongo_client, database_name=database_name, 
									   collection_name=collection_name)

		train_data, test_data = data_ingestion.initiate_data_ingestion()

		data_transformation = DataTransformation(train_data=datatrain_data, test_data=test_data)
		scaled_train_data, scaled_test_data = data_transformation.initiate_data_transformation()

		model_training = ModelTrainer(train_data=scaled_train_data, test_data=scaled_test_data)
		model_training.initiate_model_trainer()

	except Exception as e:
		raise EnergyEfficiencyException(e, sys)

