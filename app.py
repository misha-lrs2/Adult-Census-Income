import streamlit as st
import pandas as pd
import plotly.express as px
from prediction_pkg.inference_pipeline import predict_new_data

# ==========================================
# PAGE SETUP
# ==========================================
st.set_page_config(page_title="Misha's Income Predictor", layout="wide", page_icon="💸")

st.title("💸 Misha's Income Predictor")


@st.cache_data
def load_data():
    return pd.read_csv('data/adult.csv')


df = load_data()

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📖 Introduction & Ethics",
    "📊 Data Visualizations",
    "🎯 Model Performance",
    "🤖 Live Predictions"
])

# ==========================================
# TAB 1: INTRODUCTION & ETHICS
# ==========================================
with tab1:
    st.subheader("Welcome to my dashboard!")
    st.write(
        "This dashboard is built around the famous **Adult Census Income** dataset. "
        "The goal? To train a Machine Learning model that can accurately predict whether a "
        "person earns more or less than $50,000 a year. "
        "Feel free to take a look! You can view some interesting visualizations, check the model's performance, "
        "or upload data to run predictions yourself."
    )

    st.markdown("---")
    st.subheader("⚖️ The Ethical Choice")
    st.write(
        "Traditional AI models often blindly adopt all available data, which can lead to algorithmic **bias**. "
        "The original dataset contains sensitive columns such as `sex` and `race`. "
        "To prevent discrimination by the algorithm, I consciously removed these sensitive features during the training phase."
    )

# ==========================================
# TAB 2: VISUALIZATIONS (With conclusions!)
# ==========================================
with tab2:
    st.write("Explore the structure of the original dataset and discover the patterns behind financial success.")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])  # Create columns for chart and text
    with col1:
        fig_violin = px.violin(
            df, x="income", y="age", color="income", box=True,
            title="Age Distribution per Income Group",
            color_discrete_sequence=['#ff9999', '#66b3ff']
        )
        st.plotly_chart(fig_violin, use_container_width=True)
    with col2:
        st.success("**Conclusion:**")
        st.write("People with an income above 50K are generally older (the peak is around 40 years of age). "
                 "Starters on the labor market (under 30) almost always fall into the <=50K category.")

    st.markdown("---")

    col3, col4 = st.columns([2, 1])
    with col3:
        income_stats = df.groupby('education')['income'].apply(lambda x: (x == '>50K').mean()).sort_values()
        fig_bar = px.histogram(
            df, x="education", color="income", barnorm="percent",
            title="Income Percentage per Education Level",
            color_discrete_sequence=['#ff9999', '#66b3ff']
        )
        fig_bar.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': income_stats.index.tolist()})
        st.plotly_chart(fig_bar, use_container_width=True)
    with col4:
        st.success("**Conclusion:**")
        st.write("The higher the education, the greater the chance of a top income. "
                 "For masters and doctorates, the probability of earning >50K is significantly higher than with a high school diploma (HS-grad).")

    st.markdown("---")

    col5, col6 = st.columns([2, 1])
    with col5:
        occ_stats = df.groupby('occupation')['income'].apply(lambda x: (x == '>50K').mean()).sort_values()
        fig_occ = px.histogram(
            df, y="occupation", color="income", barnorm="percent", orientation="h",
            title="Probability of >50K per Occupation",
            color_discrete_sequence=['#ff9999', '#66b3ff']
        )
        fig_occ.update_layout(yaxis={'categoryorder': 'array', 'categoryarray': occ_stats.index.tolist()}, height=600)
        st.plotly_chart(fig_occ, use_container_width=True)
    with col6:
        st.success("**Conclusion:**")
        st.write(
            "Your choice of profession is one of the strongest predictors of your income. Executive roles (`Exec-managerial`) and specialized professionals (`Prof-specialty`) have by far the highest chance of a top salary.")
        st.write(
            "In contrast, the probability of >50K is minimal for people in general services (`Other-service`) or domestic help (`Priv-house-serv`). The model therefore attaches enormous value to this specific feature!")

# ==========================================
# TAB 3: MODEL PERFORMANCE
# ==========================================
with tab3:
    st.subheader("How good is the Ethical Model?")
    st.write(
        "Before we use the model live, we need to know if it can be trusted. Below you can see its performance on the test data.")

    met1, met2, met3 = st.columns(3)
    met1.metric(label="Accuracy", value="84%")
    met2.metric(label="Precision (>50K)", value="76%")
    met3.metric(label="Recall (>50K)", value="48%")

    with st.expander("ℹ️ What exactly do these scores mean?"):
        st.markdown("""
            * **Accuracy (84%):** Out of all predictions the model made (both <=50K and >50K), it was correct 84% of the time.
            * **Precision (76%):** This measures the reliability of the 'High earner' label. When the model predicts someone earns >50K, it is correct in 76% of those cases.
            * **Recall (48%):** This measures detection capability. Out of all the people who *actually* earn >50K in reality, the model successfully identified nearly half (48%). The other half were harder to recognize (e.g., they might be young or work fewer hours but still earn >50K) and were therefore missed.
            """)

    st.markdown("---")

    # --- THE CONFUSION MATRIX ---
    st.subheader("📊 Confusion Matrix")
    st.write(
        "This matrix shows exactly where the model predicted correctly on the test set, and where it made mistakes.")

    # z = [[True Negatives, False Positives],
    #      [False Negatives, True Positives]]
    z = [[4674, 240],
         [816, 750]]

    x = ['Predicted <=50K', 'Predicted >50K']
    y = ['Actual <=50K', 'Actual >50K']

    # Heatmap using Plotly
    fig_cm = px.imshow(z, text_auto=True, aspect="auto",
                       labels=dict(x="What the AI predicted", y="The reality", color="Number of people"),
                       x=x, y=y, color_continuous_scale='Blues')

    fig_cm.update_layout(title="Results on Test Data")

    # Use columns to make the matrix slightly narrower/cleaner
    col_cm1, col_cm2 = st.columns([2, 1])
    with col_cm1:
        st.plotly_chart(fig_cm, use_container_width=True)
    with col_cm2:
        st.write("**How to read this?**")
        st.write("✅ **Top-Left:** Model predicted <=50K and was correct (True Negative).")
        st.write("✅ **Bottom-Right:** Model predicted >50K and was correct (True Positive).")
        st.write("❌ **Top-Right:** Model predicted >50K but was incorrect (False Positive).")
        st.write("❌ **Bottom-Left:** Model predicted <=50K but was incorrect (False Negative).")

# ==========================================
# TAB 4: LIVE PREDICTIONS (Including Form & Error Handling)
# ==========================================
with tab4:
    st.subheader("Test the AI Model in Practice")

    # Input method selection menu
    input_method = st.radio("Choose your testing method:", ["📄 Upload a CSV file", "✍️ Enter manually"],
                            horizontal=True)

    st.markdown("---")

    # --- OPTION A: CSV UPLOAD ---
    if input_method == "📄 Upload a CSV file":
        uploaded_file = st.file_uploader("Upload your data here", type=["csv"])

        if uploaded_file is not None:
            try:
                df_uploaded = pd.read_csv(uploaded_file)

                # --- INPUT VALIDATION ---
                required_columns = ['age', 'education.num', 'occupation', 'hours.per.week']
                missing_columns = [col for col in required_columns if col not in df_uploaded.columns]

                if len(missing_columns) > 0:
                    st.error(
                        f"❌ Hold on! This doesn't seem to be valid personal data. I am missing the following columns: {', '.join(missing_columns)}")
                    st.info("Please upload a dataset that matches the Adult Census Income structure.")
                else:
                    st.success(f"File loaded successfully! ({len(df_uploaded)} rows)")

                    if st.button("Predict Incomes in Bulk", type="primary"):
                        with st.spinner('The AI is calculating...'):
                            predictions = predict_new_data(df_uploaded)
                            readable_predictions = [">50K" if v == 1 else "<=50K" for v in predictions]

                            results_df = df_uploaded.copy()
                            results_df['Predicted_Income'] = readable_predictions

                            result_columns = ['Predicted_Income'] + [col for col in results_df.columns if
                                                                       col != 'Predicted_Income']
                            results_df = results_df[result_columns]

                            st.dataframe(results_df)

                            csv_export = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv_export,
                                file_name='ai_predictions.csv',
                                mime='text/csv',
                            )
            except Exception as e:
                st.error(f"Something went wrong while reading the file. (Error: {e})")

    # --- OPTION B: MANUAL INPUT ---
    else:
        st.write("Enter the details of a fictional person to get an instant prediction.")

        # Complete mapping of all education levels to their numerical values
        edu_map = {
            "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5,
            "10th": 6, "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10,
            "Assoc-voc": 11, "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14,
            "Prof-school": 15, "Doctorate": 16
        }

        # All occupations from the Adult dataset
        all_occupations = [
            "Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial",
            "Farming-fishing", "Handlers-cleaners", "Machine-op-inspct",
            "Other-service", "Priv-house-serv", "Prof-specialty",
            "Protective-serv", "Sales", "Tech-support", "Transport-moving"
        ]

        # All work classes from the Adult dataset
        all_workclasses = [
            "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
            "Local-gov", "State-gov", "Without-pay", "Never-worked"
        ]

        # Create the form
        with st.form("manual_input_form"):
            col_a, col_b = st.columns(2)

            with col_a:
                input_age = st.slider("Age", 17, 90, 30)
                input_hours = st.slider("Hours worked per week", 1, 99, 40)
                # Automatically grab the list of keys from edu_map
                input_education = st.selectbox("Education Level", list(edu_map.keys()), index=8)  # index 8 = HS-grad
                input_edu_num = edu_map[input_education]

            with col_b:
                input_occ = st.selectbox("Occupation", all_occupations)
                input_workclass = st.selectbox("Work Sector", all_workclasses)
                input_cap_gain = st.number_input("Capital Gain", 0, 99999, 0)

            submitted = st.form_submit_button("Run Live Prediction", type="primary")

            if submitted:
                # Create a dataframe of the entered data exactly as the model expects
                new_data = pd.DataFrame([{
                    "age": input_age,
                    "education": input_education,
                    "education.num": input_edu_num,
                    "occupation": input_occ,
                    "workclass": input_workclass,
                    "capital.gain": input_cap_gain,
                    "capital.loss": 0,  # Defaulted to 0 for convenience
                    "hours.per.week": input_hours
                }])

                with st.spinner("Model is analyzing the profile..."):
                    # Send it through the pipeline!
                    prediction = predict_new_data(new_data)
                    result_text = ">50K (High Earner!)" if prediction[0] == 1 else "<=50K"

                    if prediction[0] == 1:
                        st.success(f"🎉 **Prediction:** This person likely earns **{result_text}**")
                        st.balloons()  # A little flair doesn't hurt
                    else:
                        st.warning(f"📉 **Prediction:** This person likely earns **{result_text}**")
