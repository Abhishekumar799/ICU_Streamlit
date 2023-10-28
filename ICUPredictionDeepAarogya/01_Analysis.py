from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
from PIL import Image

model = load_model('ICUPredictionDeepAarogya/Data/ICU_prediction_finetune_naive_bayes23')

cat_map = {
    "No": 0,
    "Yes": 1
}


def predict(model, input_df):
    model.memory = "ICUPredictionDeepAarogya/Data/"
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['prediction_label'][0]
    return predictions


def get_data():
    data = pd.read_csv("ICUPredictionDeepAarogya/Data/peerj-08-10337-s001.csv")
    data.columns = list(map(str.strip, list(data.columns)))
    data = data[['Procalcitonin', 'CRP', 'Ferritin', 'RR', 'LDH', 'SOB',
                 'SpO2', 'Fever', 'hypertensionhx', 'smoking_history', 'ALT',
                 'Fatigue', 'Chest.discomfort..chest.pain']]
    return data


def main():
    data = get_data()
    image2 = Image.open('ICUPredictionDeepAarogya/Images/icu.png')
    st.sidebar.info('This app is created to predict a particular patient need an ICU treatment or not. [DeepAarogya]]')
    st.sidebar.image(image2)
    st.title("ICU Prediction")

    st.sidebar.title("Check Analysis:")

    check_data = st.sidebar.checkbox('Check Decision Boundary')
    if check_data:
        st.header("Decision Boundary:")
        db = Image.open('ICUPredictionDeepAarogya/Images/sep.png')
        st.image(db)

    test_model = st.sidebar.checkbox('Test Model Online', True)
    if test_model:
        cols = data.columns
        print(cols)

        Procalcitonin = st.number_input('Procalcitonin:', min_value=data.describe()["Procalcitonin"].loc["min"],
                                        max_value=data.describe()["Procalcitonin"].loc["max"],
                                        value=data.describe()["Procalcitonin"].loc["50%"])
        CRP = st.number_input('CRP:',
                              min_value=data.describe()["CRP"].loc["min"],
                              max_value=data.describe()["CRP"].loc["max"],
                              value=data.describe()["CRP"].loc["50%"])
        Ferritin = st.number_input('Ferritin:', min_value=data.describe()["Ferritin"].loc["min"],
                                   max_value=data.describe()["Ferritin"].loc["max"],
                                   value=data.describe()["Ferritin"].loc["50%"])
        RR = st.number_input('RR:', min_value=data.describe()["RR"].loc["min"],
                             max_value=data.describe()["RR"].loc["max"],
                             value=data.describe()["RR"].loc["50%"])
        LDH = st.number_input('LDH:', min_value=data.describe()["LDH"].loc["min"],
                              max_value=data.describe()["LDH"].loc["max"],
                              value=data.describe()["LDH"].loc["50%"])
        SpO2 = st.number_input('SpO2:', min_value=data.describe()["SpO2"].loc["min"],
                               max_value=data.describe()["SpO2"].loc["max"],
                               value=data.describe()["SpO2"].loc["50%"])
        ALT = st.number_input('ALT:', min_value=data.describe()["ALT"].loc["min"],
                              max_value=data.describe()["ALT"].loc["max"],
                              value=data.describe()["ALT"].loc["50%"])
        SOB = st.selectbox('SOB:', ["No", "Yes"])
        Fever = st.selectbox('Fever:', ["No", "Yes"])
        hypertensionhx = st.selectbox('hypertensionhx:', ["No", "Yes"])
        smoking_history = st.selectbox('smoking_history:', ["No", "Yes"])
        Fatigue = st.selectbox('Fatigue:', ["No", "Yes"])
        Chestdiscomfortchestpain = st.selectbox('Chest Discomfort/Chest Pain:', ["No", "Yes"])

        output = ""

        input_dict = {
            'Procalcitonin': Procalcitonin,
            'CRP': CRP,
            'Ferritin': Ferritin,
            'RR': RR,
            'LDH': LDH,
            'SOB': cat_map[SOB],
            'SpO2': SpO2,
            'Fever': cat_map[Fever],
            'hypertensionhx': cat_map[hypertensionhx],
            'smoking_history': cat_map[smoking_history],
            'ALT': ALT,
            'Fatigue': cat_map[Fatigue],
            'Chest.discomfort..chest.pain': cat_map[Chestdiscomfortchestpain]
        }

        input_df = pd.DataFrame([input_dict])
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            print(output)
            if output == 1:
                st.warning("⚠️ Patient need to be in ICU !!!", )
            else:
                st.success("✅ Patient is fine, not recommended for ICU !!!")


if __name__ == '__main__':
    main()
