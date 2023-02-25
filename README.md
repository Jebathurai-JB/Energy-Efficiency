
# Energy Efficiency

#### Predict the cooling load and heating load in a building using machine learning.

The goal of this project is to develop a machine learning model that can accurately predict the cooling load and heating load required for a building based on its characteristics such as its relative compactness, surface area, wall area, roof area, overall height, orientation, glazing area, and glazing area distribution.

## About Dataset

performed energy analysis using 12 different building shapes simulated in Ecotect. The buildings differ with respect to the glazing area, the glazing area distribution, and the orientation, amongst other parameters. The dataset comprises 768 samples and 8 features, aiming to predict two real valued responses. 
## Models

The project uses three different machine learning models to predict the cooling load and heating load in a building:

- Random Forest Regressor
- Linear Regression
- Support Vector Machine

## Accuracy

|Models                |Heating Load|Cooling Load|
|----------------------|------------|------------|
|random forest         |99.76       |97.66       |
|linear regression     |88.73       |88.29       |
|support vector machine|18.45       |38.96       |

## Installation

- Clone the repository: `git clone https://github.com/Jebathurai-JB/Energy-Efficiency.git`
- Install the dependencies: `pip install -r requirements.txt`
    
## Usage

- To train and test the models, run `python main.py`. This will output R2 scores for the Random Forest Regressor, Linear Regressor and SVM

- To run the app, run `python app.py`. This will run Flask app in localhost
## Demo
![energy_efficiency2](https://user-images.githubusercontent.com/74975910/221373559-8662b23c-e5dc-490e-a79e-28fb5d902118.gif)

