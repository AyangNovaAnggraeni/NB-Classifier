import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load dataset untuk referensi rata-rata
df = pd.read_csv("heart_no_encod.csv")


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



# # ----------------------------
# # 6. Analytics (Visualization)
# # ----------------------------
# import matplotlib.pyplot as plt

# if st.button("Show Analytics"):
#     # Load dataset asli (belum encod)
#     df = pd.read_csv("heart_no_encod.csv")

#     # Ambil fitur numerik saja
#     num_features = [f for f, info in features_info.items() if info['type'] == 'numerical']

#     # Hitung rata-rata dataset
#     df_mean = df[num_features].mean()

#     # Ambil input user numerik
#     my_result = df_input[num_features].iloc[0]

#     # Plot side-by-side bar
#     x = range(len(num_features))
#     width = 0.35

#     fig, ax = plt.subplots(figsize=(10,5))
#     ax.bar([i - width/2 for i in x], my_result.values, width, label="My Result", color="pink")
#     ax.bar([i + width/2 for i in x], df_mean.values, width, label="Average", color="lightgreen")

#     ax.set_ylabel("Value")
#     ax.set_title("User Input vs Dataset Average (Numerical Features)")
#     ax.set_xticks(x)
#     ax.set_xticklabels(num_features, rotation=45)
#     ax.legend()

#     st.pyplot(fig)


# ----------------------------
# 6. Analytics 
# ----------------------------

st.subheader("Comparison: Your Input vs Dataset Average")

for feature in features_info.keys():
    if features_info[feature]['type'] == 'numerical':
        user_val = df_input[feature].iloc[0]
        avg_val = df[feature].mean()
        
        # Normalisasi ke skala 0–5 biar mirip contoh
        min_val, max_val = df[feature].min(), df[feature].max()
        scale_user = (user_val - min_val) / (max_val - min_val) * 5
        scale_avg = (avg_val - min_val) / (max_val - min_val) * 5
        
        st.markdown(f"**{feature}**")
        st.write(f"You: {user_val:.2f} | Avg: {avg_val:.2f}")
        
# ----------------------------
# 7. Analytics (Visualization)
# ----------------------------

# Pilih demografi
# ----------------------------
st.sidebar.header("Select up to 3 demographics")
available_demos = ["Age", "Sex"]  # bisa ditambah: Education, Cholesterol, dll
selected_demos = st.sidebar.multiselect("Choose demographics", available_demos, max_selections=3)

subsets = {}

for demo in selected_demos:
    if demo == "Sex":
        option = st.sidebar.selectbox("Select Gender", df["Sex"].unique(), key="sex")
        subsets[f"Sex: {option}"] = df[df["Sex"] == option]
    elif demo == "Age":
        min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
        age_range = st.sidebar.slider("Select Age Range", min_age, max_age, (20, 40), key="age")
        subsets[f"Age: {age_range[0]}-{age_range[1]}"] = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]
    # tambahin else-if untuk Education, Religiosity, Political ideology, dll sesuai datasetmu

# ----------------------------
# Buat comparison data
# ----------------------------
num_features = ["Age","RestingBP","Cholesterol","MaxHR","Oldpeak"]  # numerical features

compare_data = {"You": df_input[num_features].iloc[0]}

for label, subdf in subsets.items():
    compare_data[label] = subdf[num_features].mean()

compare_df = pd.DataFrame(compare_data)

# ----------------------------
# Plot bar chart
# ----------------------------
st.subheader("Compare Your Score Against Selected Groups")

fig, ax = plt.subplots(figsize=(8, 5))
compare_df.plot(kind="bar", ax=ax)
ax.set_ylabel("Average Value")
ax.set_title("Your vs Group Comparison")
st.pyplot(fig)