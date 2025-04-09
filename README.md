# paris-house-price-prediction
Random Forest from scratch for Paris house price prediction

# Project Structure
ParisHousePricePrediction.ipynb     # Main notebook 
DATA/
  └── ParisHousing.csv              # Dataset (from Kaggle)
tests.py                            # Unit tests for core functions

# Objective 
The objective of this project is to implement a Random Forest Regressor from scratch (without using sklearn.ensemble) to predict housing prices in Paris, and to compare its performance with Scikit-learn's implementation.

- Implementation of Random Forest from scratch: bootstrapping, feature selection, tree training, and prediction aggregation
- Hyperparameter tuning (number of trees, maximum depth, number of features)
- Visualization of results: prediction errors, true vs predicted prices
- Comparison with Scikit-learn's RandomForestRegressor

# Results
- Best R² (from scratch): 0.7581
- Corresponding MSE: 2.12e+12
- Sklearn R²: 0.99999
- Sklearn MSE: 1.72e+07
