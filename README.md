# Student Performance Prediction Using Machine Learning

## Project Overview
This project uses a Random Forest Regression model to estimate a student's final exam score from study hours, attendance, previous exam score, and completed assignments.

## Features
- Interactive student-detail inputs using Streamlit
- Machine Learning model built with RandomForestRegressor
- Estimated final score out of 100
- Simple pass / needs-more-preparation indicator
- Model evaluation using Mean Absolute Error (MAE) and R-squared (R²)

## Dataset
`student_data.csv` contains 500 sample records created for this educational demonstration. The data is synthetic, not real student records.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit

## How to Run
1. Install Python 3.9 or later.
2. Open a terminal in this project folder.
3. Install the libraries:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the application:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL shown in the terminal and enter student details.

## Project Workflow
1. Load the CSV dataset.
2. Select input features and target score.
3. Split data into training and testing sets.
4. Train a Random Forest Regression model.
5. Evaluate the model with MAE and R².
6. Predict a score from the entered student details.

## Important Note
This is a student learning project. The dataset is synthetic and predictions are estimates only. Real student performance depends on many additional factors, so this tool should not be used for official academic decisions.

## Author
Student Machine Learning Project
