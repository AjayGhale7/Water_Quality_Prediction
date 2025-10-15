import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("water_quality_final.pkl")

st.set_page_config(page_title="💧 Water Quality Prediction Dashboard", layout="wide")

st.title("💧 AI-Powered Water Quality Prediction")
st.markdown("### Predict whether water is **Safe (Potable)** or **Not Safe** using AI + Reference Rules")

# 📌 Reference Ranges (WHO / IS standards – approximate)
reference_ranges = {
    "ph": (6.5, 8.5),
    "Hardness": (0, 300),
    "Solids": (0, 500),
    "Chloramines": (0, 4),
    "Sulfate": (0, 400),
    "Conductivity": (0, 1500),
    "Organic_carbon": (0, 10),
    "Trihalomethanes": (0, 80),
    "Turbidity": (0, 5)
}

# Rule-based check function
def rule_based_check(values):
    for param, val in values.items():
        low, high = reference_ranges[param]
        if not (low <= val <= high):
            return False  # Out of range = unsafe
    return True

# Sidebar: Reference Ranges
st.sidebar.header("📌 Reference Ranges")
for param, (low, high) in reference_ranges.items():
    st.sidebar.write(f"**{param}:** {low} – {high}")

# Sidebar input method
st.sidebar.header("Choose Input Method")
option = st.sidebar.radio("Select input type:", ["Manual Input", "Upload CSV"])

# ✅ Manual Input
if option == "Manual Input":
    st.subheader("Enter Water Parameters")
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.01, value=7.0)
    hardness = st.number_input("Hardness", min_value=0.0, step=0.01, value=150.0)
    solids = st.number_input("Solids (ppm)", min_value=0.0, step=1.0, value=200.0)
    chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, step=0.01, value=2.0)
    sulfate = st.number_input("Sulfate (ppm)", min_value=0.0, step=0.01, value=250.0)
    conductivity = st.number_input("Conductivity (µS/cm)", min_value=0.0, step=0.01, value=400.0)
    organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, step=0.01, value=8.0)
    trihalomethanes = st.number_input("Trihalomethanes (µg/L)", min_value=0.0, step=0.01, value=60.0)
    turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, step=0.01, value=2.0)

    # Predict button
    if st.button("🔮 Predict"):
        input_data = pd.DataFrame([[ph, hardness, solids, chloramines, sulfate,
                                    conductivity, organic_carbon, trihalomethanes, turbidity]],
                                  columns=['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                                           'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'])
        
        # Convert row to dict for rule-based check
        values_dict = dict(zip(input_data.columns, input_data.iloc[0]))

        # 1️⃣ Model prediction
        model_pred = model.predict(input_data)[0]

        # 2️⃣ Rule-based override
        safe_by_rules = rule_based_check(values_dict)

        if safe_by_rules and model_pred == 1:
            st.success("✅ The water is **SAFE for drinking** (Potable)")
        else:
            st.error("⚠️ The water is **NOT SAFE for drinking** (Not Potable)")

        # 🔍 Comparison Table with Reference Ranges
        st.subheader("Comparison with Reference Ranges")
        comparison_data = []
        for param, value in zip(input_data.columns, input_data.iloc[0]):
            low, high = reference_ranges[param]
            if low <= value <= high:
                status = "✅ Within Range"
            else:
                status = f"⚠️ Out of Range (Expected {low}–{high})"
            comparison_data.append([param, value, f"{low} – {high}", status])

        comparison_df = pd.DataFrame(comparison_data, columns=["Parameter", "Your Value", "Reference Range", "Status"])
        st.dataframe(comparison_df)

# ✅ CSV Upload
elif option == "Upload CSV":
    st.subheader("Upload a CSV file with water quality data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("📊 Uploaded Data Preview:")
        st.dataframe(data.head())

        if st.button("🔮 Predict for Uploaded Data"):
            results = []
            for _, row in data.iterrows():
                values_dict = row.to_dict()
                model_pred = model.predict(pd.DataFrame([row]))[0]
                safe_by_rules = rule_based_check(values_dict)
                final_pred = "Safe" if (safe_by_rules and model_pred == 1) else "Not Safe"
                results.append(final_pred)

            data['Prediction'] = results
            st.success("✅ Predictions Completed!")
            st.dataframe(data)

            # Download option
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Predictions", csv, "water_quality_predictions.csv", "text/csv")

st.markdown("---")
st.info("Developed with ❤️ for AI-based Water Quality Monitoring")
