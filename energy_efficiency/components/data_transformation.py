import sys



class DataTransformation:
	def __init__(self, train_data, test_data):
		try:
			self.train_data = train_data
			self.test_data = test_data

		except Exception as e:
			raise EnergyEfficiencyException(e, sys)

	def feature_scaling(self, train_data):
		scaler = StandardScaler()
		scaler.fit



#################################   YET  TO  FINISH   ###########################################