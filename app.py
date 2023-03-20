import pickle
import numpy as np
import pandas as pd
from flask import Flask, url_for, render_template, request

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():

	if request.method == 'POST':
		
		data_form = request.form
		data = [data_form[i] for i in data_form]
		data = np.array(data[:-1])
		data = data.reshape(1, -1)

		heating_metrics = pd.read_csv('heating_metrics.csv', index_col='Unnamed: 0')
		cooling_metrics = pd.read_csv('cooling_metrics.csv', index_col='Unnamed: 0')

		heating_load_model_name = heating_metrics['Accuracy'].idxmax()
		cooling_load_model_name = cooling_metrics['Accuracy'].idxmax()
		

		cooling_button = request.form.get('cooling_button')

		if cooling_button is not None:
			cooling_load_model = pickle.load(open(f'saved models/{cooling_load_model_name}/cooling_load_model.pkl', 'rb'))
			prediction = cooling_load_model.predict(data)
			button = 'cooling'
			
		else:
			heating_load_model = pickle.load(open(f'saved models/{heating_load_model_name}/heating_load_model.pkl', 'rb'))
			prediction = heating_load_model.predict(data)

			button = None

		return render_template('home.html', prediction=prediction[0], button=button)

	else:
		return render_template('home.html')


if __name__ == '__main__':
	app.run(debug=True)