import numpy as np
from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
from PIL import Image

model = load_model('ICUPredictionDeepAarogya/Data/ICU_prediction_finetune_xgb')

cat_map = {
    "No": 0,
    "Yes": 1,
    "Not Available": np.nan,
    "Male": 1,
    "Female": 0
}


def predict(model, input_df):
    model.memory = "ICUPredictionDeepAarogya/Data/"
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['prediction_label'][0]
    confidence = predictions_df['prediction_score'][0]
    return predictions, confidence


def get_data():
    data = pd.read_csv("ICUPredictionDeepAarogya/Data/peerj-08-10337-s001.csv")
    data.columns = list(map(str.strip, list(data.columns)))
    data = data[['Age', 'Gender..female.0..male1.', 'Fever', 'Cough', 'SOB', 'Fatigue', 'Sputum', 'Myalgia', 'Diarrhea',
             'Nausea.Vomiting', 'Sore.throat', 'Chest.discomfort..chest.pain', 'smoking_history',
             'hypertensionhx', 'diabeteshx', 'coronaryheartdiseasehx', 'copdhx', 'carcinomahx', 'ckdhx', 'ALT',
            'HR', 'Lymphocyte', 'SpO2', 'Procalcitonin', 'RR', 'Systolic.BP', 'Temperature']]
    return data


def main():
    data = get_data()
    image2 = Image.open('ICUPredictionDeepAarogya/Images/icu.png')
    st.sidebar.info('This app is created to predict a particular patient need ICU treatment or no. [DeepAarogya]] - Version 2')
    st.sidebar.image(image2)
    st.title("ICU Prediction V2")

    st.sidebar.title("Check Analysis:")

    check_data = st.sidebar.checkbox('Check Feature Importance')
    if check_data:
        st.header("Feature Importance:")
        db = Image.open('ICUPredictionDeepAarogya/Images/ft_importance.png')
        st.image(db)

    test_model = st.sidebar.checkbox('Test Model V2 Online', True)
    if test_model:
        cols = data.columns
        print(cols)

        # Index(['Age', 'Gender..female.0..male1.', 'Fever', 'Cough', 'SOB', 'Fatigue',
        #        'Sputum', 'Myalgia', 'Diarrhea', 'Nausea.Vomiting', 'Sore.throat',
        #        'Chest.discomfort..chest.pain', 'smoking_history', 'hypertensionhx',
        #        'diabeteshx', 'coronaryheartdiseasehx', 'copdhx', 'carcinomahx',
        #        'ckdhx', 'ALT', 'HR', 'Lymphocyte', 'SpO2', 'Procalcitonin', 'RR',
        #        'Systolic.BP', 'Temperature']

        if st.checkbox("Do you have patient Age?", False):
            Age = st.number_input('Age:', min_value=data.describe()["Age"].loc["min"],
                                        max_value=data.describe()["Age"].loc["max"],
                                        value=data.describe()["Age"].loc["50%"])
        else:
            Age = np.nan

        Gender = st.selectbox('Gender:', ["Not Available", "Male", "Female"])
        Fever = st.selectbox('Fever:', ["Not Available", "No", "Yes"])
        Cough = st.selectbox('Cough:', ["Not Available", "No", "Yes"])
        SOB = st.selectbox('SOB:', ["Not Available", "No", "Yes"])
        Fatigue = st.selectbox('Fatigue:', ["Not Available", "No", "Yes"])
        Sputum = st.selectbox('Sputum:', ["Not Available", "No", "Yes"])
        Myalgia = st.selectbox('Myalgia:', ["Not Available", "No", "Yes"])
        Diarrhea = st.selectbox('Diarrhea:', ["Not Available", "No", "Yes"])
        Nausea = st.selectbox('Nausea/Vomiting:', ["Not Available", "No", "Yes"])
        Sore_throat = st.selectbox('Sore throat:', ["Not Available", "No", "Yes"])
        Chest_discomfort_chest_pain = st.selectbox('Chest.discomfort/chest pain:', ["Not Available", "No", "Yes"])
        smoking_history = st.selectbox('Smoking History:', ["Not Available", "No", "Yes"])
        hypertensionhx = st.selectbox('Hypertension history:', ["Not Available", "No", "Yes"])
        diabeteshx = st.selectbox('Diabetes History:', ["Not Available", "No", "Yes"])
        coronaryheartdiseasehx = st.selectbox('Coronary Heart Disease History:', ["Not Available", "No", "Yes"])
        copdhx = st.selectbox('COPD History:', ["Not Available", "No", "Yes"])
        carcinomahx = st.selectbox('Carcinoma History:', ["Not Available", "No", "Yes"])
        ckdhx = st.selectbox('ckd History:', ["Not Available", "No", "Yes"])


        if st.checkbox("Do you have patient ALT?", False):
            ALT = st.number_input('ALT:', min_value=data.describe()["ALT"].loc["min"],
                                        max_value=data.describe()["ALT"].loc["max"],
                                        value=data.describe()["ALT"].loc["50%"])
        else:
            ALT = np.nan

        if st.checkbox("Do you have patient HR?", False):
            HR = st.number_input('HR:', min_value=data.describe()["HR"].loc["min"],
                                        max_value=data.describe()["HR"].loc["max"],
                                        value=data.describe()["HR"].loc["50%"])
        else:
            HR = np.nan

        if st.checkbox("Do you have patient Lymphocyte?", False):
            Lymphocyte = st.number_input('Lymphocyte:', min_value=data.describe()["Lymphocyte"].loc["min"],
                                        max_value=data.describe()["Lymphocyte"].loc["max"],
                                        value=data.describe()["Lymphocyte"].loc["50%"])
        else:
            Lymphocyte = np.nan

        if st.checkbox("Do you have patient SpO2?", False):
            SpO2 = st.number_input('SpO2:', min_value=data.describe()["SpO2"].loc["min"],
                                        max_value=data.describe()["SpO2"].loc["max"],
                                        value=data.describe()["SpO2"].loc["50%"])
        else:
            SpO2 = np.nan

        if st.checkbox("Do you have patient Procalcitonin?", False):
            Procalcitonin = st.number_input('Procalcitonin:', min_value=data.describe()["Procalcitonin"].loc["min"],
                                        max_value=data.describe()["Procalcitonin"].loc["max"],
                                        value=data.describe()["Procalcitonin"].loc["50%"])
        else:
            Procalcitonin = np.nan

        if st.checkbox("Do you have patient RR?", False):
            RR = st.number_input('RR:', min_value=data.describe()["RR"].loc["min"],
                                        max_value=data.describe()["RR"].loc["max"],
                                        value=data.describe()["RR"].loc["50%"])
        else:
            RR = np.nan

        if st.checkbox("Do you have patient Systolic BP?", False):
            Systolic_BP = st.number_input('Systolic BP:', min_value=data.describe()["Systolic.BP"].loc["min"],
                                        max_value=data.describe()["Systolic.BP"].loc["max"],
                                        value=data.describe()["Systolic.BP"].loc["50%"])
        else:
            Systolic_BP = np.nan

        if st.checkbox("Do you have patient Temperature?", False):
            Temperature = st.number_input('Temperature:', min_value=data.describe()["Temperature"].loc["min"],
                                          max_value=data.describe()["Temperature"].loc["max"],
                                          value=data.describe()["Temperature"].loc["50%"])
        else:
            Temperature = np.nan

        output = ""

        input_dict = {
            'Age': Age,
            'Gender..female.0..male1.': cat_map[Gender],
            'Fever': cat_map[Fever],
            'Cough': cat_map[Cough],
            'SOB': cat_map[SOB],
            'Fatigue': cat_map[Fatigue],
            'Sputum': cat_map[Sputum],
            'Myalgia': cat_map[Myalgia],
            'Diarrhea': cat_map[Diarrhea],
            'Nausea.Vomiting': cat_map[Nausea],
            'Sore.throat': cat_map[Sore_throat],
            'Chest.discomfort..chest.pain': cat_map[Chest_discomfort_chest_pain],
            'smoking_history': cat_map[smoking_history],
            'hypertensionhx': cat_map[hypertensionhx],
            'diabeteshx': cat_map[diabeteshx],
            'coronaryheartdiseasehx': cat_map[coronaryheartdiseasehx],
            'copdhx': cat_map[copdhx],
            'carcinomahx': cat_map[carcinomahx],
            'ckdhx': cat_map[ckdhx],
            'ALT': ALT,
            'HR': HR,
            'Lymphocyte': Lymphocyte,
            'SpO2': SpO2,
            'Procalcitonin': Procalcitonin,
            'RR': RR,
            'Systolic.BP': Systolic_BP,
            'Temperature': Temperature
        }


        input_df = pd.DataFrame([input_dict])
        if st.button("Predict"):
            output, confidence = predict(model=model, input_df=input_df)
            print(output)
            if output == 1:
                st.warning(f"⚠️ Patient need to be in ICU !!! (Confidence = {confidence*100} %)", )
            else:
                st.success(f"✅ Patient is fine, not recommended for ICU !!! (Confidence = {confidence*100} %)")


if __name__ == '__main__':
    main()
