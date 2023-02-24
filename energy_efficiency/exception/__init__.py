import os
import sys

class EnergyEfficiencyException(Exception):

	def __init__(self, error_message, error_detail):
		super().__init__(error_message)
		self.error_message = EnergyEfficiencyException.error_message_detail(error_message, error_detail)


	@staticmethod
	def error_message_detail(error, error_detail):
		_, _, exc_tb = error_detail.exc_info()
		line_number = exc_tb.tb_frame.f_lineno

		file_name = exc_tb.tb_frame.f_code.co_filename

		error_message = f"Error occurred python script name [{file_name}]" \
                        f" line number [{exc_tb.tb_lineno}] error message [{error}]."
                        
		return error_message


	def __str__(self):
		return self.error_message


	def __repr__(self):
		return EnergyEfficiencyException.__name__.__str__()