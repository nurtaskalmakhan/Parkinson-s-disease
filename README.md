# Parkinson-s-disease

# Parkinson's Disease Prediction

This project focuses on predicting Parkinson's disease using the Parkinson's Disease Classification dataset. The goal is to train an XGBoost classifier to accurately predict the presence or absence of Parkinson's disease based on various features.

## Dataset

The dataset used for this project is "parkinsons.data". It contains various clinical measurements of voice signals and other relevant features of individuals, along with their Parkinson's disease status.

## Installation

To run this project, you need to have Python installed. Additionally, the following libraries are required:

- pandas
- numpy
- xgboost
- scikit-learn
- seaborn

You can install these dependencies using pip:

pip install pandas numpy xgboost scikit-learn seaborn


## Usage

1. Clone the repository:

git clone [Parkinson-s-disease]


2. Navigate to the project directory:

cd parkinsons-disease-prediction


3. Make sure the dataset file "parkinsons.data" is placed in the project directory.

4. Run the Python script:

python parkinsons_disease_prediction.py


5. The script will perform the following tasks:
   - Load and preprocess the dataset
   - Split the dataset into training and testing sets
   - Scale the features using MinMaxScaler
   - Train an XGBoost classifier on the training set
   - Make predictions on the testing set
   - Print the predicted labels and the actual labels
   - Display a classification report showing the precision, recall, and F1-score of the predictions

6. The classification report will provide insights into the performance of the model.

## Contributing

Contributions to this project are welcome. Feel free to submit bug reports, feature requests, or pull requests to help improve the project.

Please follow the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md) for more details on how to contribute.

## License

This project is licensed under the [MIT License](LICENSE).
