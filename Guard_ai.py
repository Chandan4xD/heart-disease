import streamlit as st
import pandas as pd
import pickle

st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #28a745;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

model = pickle.load(open("heart_model.pkl", "rb"))

# Sidebar
st.sidebar.title("❤️ CardioGuard")

page = st.sidebar.selectbox(
    "Navigation",
    ["Prediction", "BMI Calculator", "About"]
)

st.sidebar.success("Model Accuracy: 98.5%")

st.warning(
    "This tool is for educational purposes only and does not replace professional medical advice."
)

if page == "Prediction":

    st.title("❤️ CardioGuard")
    st.subheader("Heart Disease Prediction System")

    patientid = st.number_input("Patient ID", 1, 10000, 1)
    age = st.number_input("Age", 1, 100, 40)
    gender_option = st.selectbox("Gender", ["Male", "Female"])
    gender = 1 if gender_option == "Male" else 0
    chestpain = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    restingBP = st.number_input("Resting BP", 50, 250, 120)
    serumcholestrol = st.number_input("Serum Cholesterol", 100, 600, 200)
    fastingbloodsugar = st.selectbox("Fasting Blood Sugar", [0, 1])
    restingrelectro = st.selectbox("Resting ECG", [0, 1, 2])
    maxheartrate = st.number_input("Max Heart Rate", 50, 250, 150)
    exerciseangia = st.selectbox("Exercise Angina", [0, 1])
    oldpeak = st.number_input("Old Peak", 0.0, 10.0, 1.0)

    # FIX: Slope now 1-3 (Cleveland standard) instead of 0-3
    slope_option = st.selectbox(
        "Slope",
        [1, 2, 3],
        format_func=lambda x: {
            1: "1 - Upsloping (Healthy)",
            2: "2 - Flat (Moderate)",
            3: "3 - Downsloping (Concerning)"
        }[x]
    )
    slope = slope_option

    noofmajorvessels = st.selectbox("No. of Major Vessels", [0, 1, 2, 3])

    if st.button("Predict"):

        input_data = pd.DataFrame([[
            age,
            gender,
            chestpain,
            restingBP,
            serumcholestrol,
            fastingbloodsugar,
            restingrelectro,
            maxheartrate,
            exerciseangia,
            oldpeak,
            slope,
            noofmajorvessels
        ]], columns=[
            'age', 'gender', 'chestpain', 'restingBP', 'serumcholestrol',
            'fastingbloodsugar', 'restingrelectro', 'maxheartrate',
            'exerciseangia', 'oldpeak', 'slope', 'noofmajorvessels'
        ])

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1] * 100
        health_score = 100 - probability

        st.metric("Heart Disease Risk", f"{probability:.2f}%")
        st.metric("Heart Health Score", f"{health_score:.0f}/100")

        if prediction[0] == 1:
            st.error("⚠ High Risk of Heart Disease")
            st.subheader("Recommendations")
            st.write("• Exercise 30 minutes daily")
            st.write("• Eat healthy food")
            st.write("• Avoid smoking")
            st.write("• Reduce cholesterol")
            st.write("• Consult a cardiologist")
        else:
            st.success("✅ Low Risk of Heart Disease")

        report = f"""
CARDIOGUARD REPORT

Risk Percentage: {probability:.2f}%
Health Score: {health_score:.0f}/100
Prediction: {'High Risk' if prediction[0] == 1 else 'Low Risk'}
"""

        st.download_button(
            "📥 Download Report",
            report,
            file_name="heart_report.txt"
        )

elif page == "BMI Calculator":

    st.title("🏃 BMI Calculator")

    weight = st.number_input("Weight (kg)", 1.0, 300.0, 70.0)
    height = st.number_input("Height (m)", 0.5, 2.5, 1.70)

    if st.button("Calculate BMI"):
        bmi = weight / (height ** 2)
        st.metric("BMI", f"{bmi:.2f}")

        if bmi < 18.5:
            st.warning("Underweight")
        elif bmi < 25:
            st.success("Normal Weight")
        elif bmi < 30:
            st.warning("Overweight")
        else:
            st.error("Obese")

elif page == "About":

    st.title("ℹ About CardioGuard")

    st.write("""
    CardioGuard is a Machine Learning based
    Heart Disease Prediction System.

    It helps estimate the likelihood of heart
    disease using patient health information.
    """)

    st.subheader("🎯 Project Features")
    st.write("✔ Heart Disease Prediction")
    st.write("✔ Risk Percentage")
    st.write("✔ Health Score")
    st.write("✔ BMI Calculator")
    st.write("✔ Download Report")
    st.write("✔ Recommendations")

    st.subheader("📊 Dataset Information")
    st.write("Total Features: 13")
    st.write("Algorithm: Random Forest")
    st.write("Target Variable: Heart Disease")

    st.subheader("❤️ Healthy Heart Tips")
    st.info("Exercise regularly")
    st.info("Eat fruits and vegetables")
    st.info("Avoid smoking")
    st.info("Control blood pressure")
    st.info("Reduce stress")

    st.subheader("🛠 Technologies Used")
    st.write("• Python")
    st.write("• Streamlit")
    st.write("• Pandas")
    st.write("• Scikit-Learn")
    st.write("• Machine Learning")

    st.subheader("👨‍💻 Developed By")
    st.write("1. Arpan Das")
    st.write("2. Chandan Kumar Mishra")
    st.write("3. MD. Belal")

    st.subheader("Project Guide")
    st.write("Himadri Bushan Mahapatra")

    st.write("CardioGuard Project")
