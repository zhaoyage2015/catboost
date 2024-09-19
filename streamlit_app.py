
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# 加载已保存的模型
model = joblib.load('CatBoost.pkl')

# 设置标题
st.title("RF Prediction")

# 定义特征名称
feature_names = ['Age', 'Tbil', 'Neutrophils', 'Baseline DBP', 'MLS', 'Initial NIHSS', 
                 'Hemorrhagic transformation', 'Serum glucose', 'Hypertension', 'ASPECTS', 
                 'Pulmonary infection', 'NLR']

# 用户输入部分
NLR = st.number_input("NLR:", min_value=1.00, max_value=100.00, value=10.00)
PI = st.selectbox("Pulmonary infection (0=NO, 1=Yes):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'Yes (1)')
ASPECTS = st.number_input("ASPECTS:", min_value=5, max_value=10, value=8)
HTN = st.selectbox("Hypertension (0=NO, 1=Yes):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'Yes (1)')
GLU = st.number_input("Serum glucose:", min_value=2.2, max_value=32.0, value=8.0)
HT = st.selectbox("Hemorrhagic transformation (0=NO, 1=Yes):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'Yes (1)')
NIHSS = st.number_input("Initial NIHSS:", min_value=3, max_value=42, value=16)
MLS = st.number_input("MLS:", min_value=0.00, max_value=30.00, value=2.88)
DBP = st.number_input("Baseline DBP:", min_value=40, max_value=160, value=85)
NE = st.number_input("Neutrophils:", min_value=1.50, max_value=30.00, value=8.00)
TB = st.number_input("Tbil:", min_value=3.00, max_value=100.00, value=17.00)
Age = st.number_input("Age:", min_value=18, max_value=100, value=66)

# 收集用户输入的数据
feature_values = [Age, TB, NE, DBP, MLS, NIHSS, HT, GLU, HTN, ASPECTS, PI, NLR]
features = np.array([feature_values])

# 预测与结果展示
if st.button("Predict"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")
    probability = predicted_proba[predicted_class] * 100
    st.write(f"**Predicted Probability:** {probability:.2f}%")

    # 使用 SHAP 解释模型
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    # 显示 SHAP force_plot
    shap.initjs()  # 初始化 SHAP 的 JavaScript 依赖
    fig, ax = plt.subplots()
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True, show=False)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
