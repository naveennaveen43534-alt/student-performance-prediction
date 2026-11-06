import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")
st.title("🎓 Student Performance Prediction")
st.write("A beginner-friendly Machine Learning project that estimates a student's final exam score.")

@st.cache_data
def load_data():
    return pd.read_csv("student_data.csv")

@st.cache_resource
def train_model():
    data = load_data()
    features = ["study_hours_per_day", "attendance_percentage",
                "previous_exam_score", "assignments_completed"]
    X = data[features]
    y = data["final_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, mean_absolute_error(y_test, predictions), r2_score(y_test, predictions)

try:
    model, mae, r2 = train_model()
    st.subheader("Enter student details")
    study = st.slider("Study hours per day", 0.0, 12.0, 3.0, 0.5)
    attendance = st.slider("Attendance (%)", 0, 100, 75)
    previous = st.slider("Previous exam score", 0, 100, 60)
    assignments = st.slider("Assignments completed (out of 10)", 0, 10, 7)

    if st.button("Predict final score", type="primary"):
        input_data = pd.DataFrame([{
            "study_hours_per_day": study,
            "attendance_percentage": attendance,
            "previous_exam_score": previous,
            "assignments_completed": assignments
        }])
        prediction = float(model.predict(input_data)[0])
        prediction = max(0, min(100, prediction))
        st.metric("Predicted final score", f"{prediction:.1f} / 100")
        if prediction >= 50:
            st.success("Prediction: Likely to pass")
        else:
            st.warning("Prediction: Needs more preparation")

    st.caption(f"Model evaluation on test data — MAE: {mae:.2f}; R²: {r2:.2f}")
    st.info("This is an educational demonstration using a sample synthetic dataset. It is not an official assessment tool.")
    with st.expander("View sample dataset"):
        st.dataframe(load_data().head(20), use_container_width=True)
except FileNotFoundError:
    st.error("student_data.csv was not found. Keep it in the same folder as app.py.")
