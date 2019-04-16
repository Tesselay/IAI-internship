# IAI-internship
Applications/Files created in my 2 week internship at the Institute for Artificial Intelligence in Bremen (http://ai.uni-bremen.de/).

DTWMovement
- Includes python file that can classify movements into run, fall or jump. Includes a few example datasets recorded in Unreal Engine (see https://github.com/Tesselay/IAI-internship/tree/master/DTWProject_cpp_ue/Source). Has function for plotting and individualizable testing parameters.

DTWProject_cpp_ue
- Unreal Engine C++ Source files that were used to record movements. Barebone example arena with added features of recording movements on demand and writing xyz location, velocity and acceleration, as well as tick, time per tick and time total to an csv-file. 

Diverse
- DesicionTree.py: Includes Decision Tree Classifier Example based on the Iris Flower Dataset as well as an own written DT model. (Only own model works, example needs to be revised)
- GridSearch.py: A file that does GridSearch for sklearn DT, Support Vector Classifier and XGBoost and plots a Heatmap based on it. Functions written to make GridSearch as customizable and efficient as possible. 
- Testing.py: Includes sklearn's DT Classifier and KNN for the Iris Flower Dataset, as well as two plots for visualization. 
- XGBoost.py: Basic testing of XGBoost with the Iris Flower Dataset.

Winequality_regression
- Basic testing with the Winequality regression dataset (included), uses sklearn Linear Regression Model and has functions for ols-cost and gradient_descent that have yet to be completed. Calculates mean squared error.
