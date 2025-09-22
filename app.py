import streamlit as st
import pandas as pd
import pickle

# ----------------------------
# 1. Load model & columns
# ----------------------------
with open("nb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    train_cols = pickle.load(f)

st.title("Heart Disease Prediction")

# ----------------------------
# 2. Define feature info
# ----------------------------
features_info = {
    'Age': {'type': 'numerical', 'min': 0, 'max': 120},
    'Sex': {'type': 'categorical', 'options': ['F', 'M']},
    'ChestPainType': {'type': 'categorical', 'options': ['ASY', 'ATA', 'NAP', 'TA']},
    'RestingBP': {'type': 'numerical', 'min': 50, 'max': 250},
    'Cholesterol': {'type': 'numerical', 'min': 100, 'max': 600},
    'FastingBS': {'type': 'categorical', 'options': ['N', 'Y']},  # will convert to 0/1
    'RestingECG': {'type': 'categorical', 'options': ['LVH', 'Normal', 'ST']},
    'MaxHR': {'type': 'numerical', 'min': 60, 'max': 250},
    'ExerciseAngina': {'type': 'categorical', 'options': ['N', 'Y']},
    'Oldpeak': {'type': 'numerical', 'min': 0.0, 'max': 10.0},
    'ST_Slope': {'type': 'categorical', 'options': ['Down', 'Flat', 'Up']}
}

# ----------------------------
# 3. Collect input
# ----------------------------
input_data = {}
for feature, info in features_info.items():
    if info['type'] == 'numerical':
        if feature == 'Oldpeak':
            input_data[feature] = st.number_input(
                f"{feature} (decimal allowed)", info['min'], info['max'], 0.0, 0.1
            )
        else:
            input_data[feature] = st.number_input(
                feature, info['min'], info['max'], info['min']
            )
    else:
        input_data[feature] = st.selectbox(feature, info['options'])

# ----------------------------
# 4. Preprocess input
# ----------------------------
df_input = pd.DataFrame([input_data])

# Convert FastingBS to numeric
df_input['FastingBS'] = df_input['FastingBS'].map({'N': 0, 'Y': 1})

# One-hot encode other categorical features
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
df_encoded = pd.get_dummies(df_input, columns=categorical_cols, drop_first=False)

# Match columns with training data
df_encoded = df_encoded.reindex(columns=train_cols, fill_value=0)

# ----------------------------
# 5. Prediction
# ----------------------------
if st.button("Predict"):
    prediction = model.predict(df_encoded)
    prob = model.predict_proba(df_encoded)

    st.subheader("Prediction Result")
    st.write("💓 Heart Disease" if prediction[0] == 1 else "✅ No Heart Disease")

    st.subheader("Prediction Probabilities")
    st.write(f"Probability of Heart Disease: {prob[0][1]*100:.2f}%")
    st.write(f"Probability of No Heart Disease: {prob[0][0]*100:.2f}%")
