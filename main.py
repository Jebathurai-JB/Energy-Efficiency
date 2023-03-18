import os
import sys
import pymongo
from energy_efficiency.components.data_ingestion import DataIngestion
from energy_efficiency.components.model_training import ModelTrainer
from energy_efficiency.exception import EnergyEfficiencyException

database_name = "Ineuron_project"
collection_name = "Energy_Efficiency"
mongo_client = pymongo.MongoClient("mongodb+srv://jebathurai:<password>@cluster0.irlqnsu.mongodb.net/?retryWrites=true&w=majority")


if __name__ == '__main__':

	try:
		data_ingestion = DataIngestion(mongo_client=mongo_client, database_name=database_name, 
									   collection_name=collection_name)

		train_df, test_df = data_ingestion.initiate_data_ingestion()


		model_training = ModelTrainer(train_df=train_df, test_df=test_df)
		model_training.initiate_model_trainer()

	except Exception as e:
		raise EnergyEfficiencyException(e, sys)

