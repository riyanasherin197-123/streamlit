import pandas as pd
import streamit as st
import joblib

# Load the pre-trained model
model = joblib.load('model.pk1')
# load scaler and encoder
scaler = joblib.load('scaler.pk1')
encoder = joblib.load('encoder.pk1')


st.title("students Performance Prediction")
st.write("Enter the details of the student to predict their performance.")

# Input fields
name=st.text_input("Student Name", value="enter the  name", help="Enter the name of the student")
hours_studied = st.number_input("Hours Studied", min_value=0, max_value=10, help="Enter the number of hours the student has studied")
previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, help="Enter the previous scores of the student")
Extracurricular_activities = st.selectbox("Extracurricular Activities", options=["Yes", "No"], help="Does the student participate in extracurricular activities?")
eca=encoder.transform([[Extracurricular_activities]])
sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=12, help="Enter the average number of hours the student sleeps per night")
SQpapers = st.number_input("Sample Question Papers Practice", min_value=0, max_value=20, help="Enter the number of sample question papers practiced by the student")

# Prepare the input data
input_data = pd.DataFrame({"Name": [name],
    "Hours Studied": [hours_studied],
                           "Previous Scores": [previous_scores],
                           "Extracurricular Activities": eca[0],
                            "Sleep Hours": [sleep_hours],
                            "Sample Question Papers Practiced": [SQpapers]})

st.write(input_data)

# Scale the input data
scaled_data = scaler.transform(input_data.iloc[:,1:])
st.write(scaled_data)

# Make prediction
if st.button("Predict Performance"):
    prediction = model.predict(scaled_data)
    if prediction[0]>=85:
        st.success(f"The predicted performance score for {name} is: {prediction[0]:.2f}")
        st.balloons()
    elif prediction[0]>=60:
        st.warning(f"The predicted performance score for {name} is: {prediction[0]:.2f}. Needs Improvement.")
    else:
        st.error(f"The predicted performance score for {name} is: {prediction[0]:.2f}. Poor Performance.")