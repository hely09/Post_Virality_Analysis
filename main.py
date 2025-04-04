import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px


# Configure page layout
st.set_page_config(
    page_title="Post Virality Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar with navigation menu
with st.sidebar:
    st.title("Post Virality Analysis")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "Analytics", "Prediction"],
        icons=["speedometer2", "graph-up", "magic"],
        default_index=0,
    )

# Load data
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("File uploaded successfully!")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.session_state.df = None
else:
    st.session_state.df = None

# Ensure target column exists
if st.session_state.df is not None:
    if 'viral' not in st.session_state.df.columns:
        st.error("Your dataset must contain a 'viral' column (1=viral, 0=not viral) for predictions")
        st.session_state.model = None
        st.session_state.target_exists = False
    else:
        df = st.session_state.df.copy()
        df.dropna(inplace=True)
        label_encoders = {}

        X = df.drop(columns=['viral'])
        y = df['viral']

        for col in X.select_dtypes(include='object').columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        st.session_state.model = model
        st.session_state.encoders = label_encoders
        st.session_state.features = X.columns.tolist()
        st.session_state.target_exists = True
else:
    st.session_state.model = None
    st.session_state.target_exists = False

# Dashboard
if selected == "Dashboard":
    st.header("Dashboard")
    if st.session_state.df is not None:
        st.subheader("Dataset Preview (First 5 Rows)")
        st.dataframe(st.session_state.df.head())

        st.subheader("Dataset Shape")
        rows, cols = st.session_state.df.shape
        st.metric("Total Rows", rows)
        st.metric("Total Columns", cols)

        st.subheader("Quick Stats")
        st.write(f"Missing Values: {st.session_state.df.isna().sum().sum()}")
        st.write(f"Duplicate Rows: {st.session_state.df.duplicated().sum()}")
    else:
        st.warning("Please upload a CSV file to view the dashboard")

# Analytics
elif selected == "Analytics":
    st.header("Analytics")
    df = st.session_state.df
    if df is not None:
        if 'text_content_type' in df.columns:
            type_counts = df['text_content_type'].value_counts().reset_index()
            type_counts.columns = ['Content Type', 'Count']
            bar_fig = px.bar(
                type_counts,
                x='Content Type',
                y='Count',
                title="Count of Post Content Types"
            )
            st.plotly_chart(bar_fig)

        if 'platform' in df.columns:
            pie_fig = px.pie(
                df,
                names='platform',
                title='Platform Distribution',
                hole=0.4
            )
            st.plotly_chart(pie_fig)

        metrics = ['likes', 'shares', 'comments']
        available_metrics = [m for m in metrics if m in df.columns]

        if available_metrics:
            box_fig = px.box(
                df,
                y=available_metrics,
                title='Distribution of Likes, Shares, and Comments'
            )
            st.plotly_chart(box_fig)

        if all(col in df.columns for col in ['previous_engagement', 'virality_score', 'viral']):
            st.subheader("Scatter Plot: Previous Engagement vs Virality Score")
            scatter_fig = px.scatter(
                df,
                x='previous_engagement',
                y='virality_score',
                color=df['viral'].astype(str),
                title='Previous Engagement vs Virality Score',
                labels={'color': 'Viral Status'},
                hover_data=['meme_id', 'likes', 'shares', 'comments'] if 'meme_id' in df.columns else None
            )
            st.plotly_chart(scatter_fig)

            # Top 10 Posts with Highest Virality Score
        if all(col in df.columns for col in
               ['meme_id', 'text_content_type', 'platform', 'likes', 'shares', 'comments', 'virality_score']):
            st.subheader("Top 10 Posts with Highest Virality Score")
            top_posts = df[
                ['meme_id', 'text_content_type', 'platform', 'likes', 'shares', 'comments', 'virality_score']] \
                .sort_values(by='virality_score', ascending=False).head(10)
            st.dataframe(top_posts)

            # Average Engagement by Platform
        if 'platform' in df.columns:
            st.subheader("Average Engagement by Platform")
            engagement_metrics = ['likes', 'shares', 'comments', 'virality_score']
            engagement_avg = df.groupby('platform')[engagement_metrics].mean().reset_index()
            st.dataframe(engagement_avg)

    else:
            st.warning("Please upload a CSV file to view analytics")

# Prediction
elif selected == "Prediction":
    st.header("Prediction")
    if st.session_state.df is not None and st.session_state.model is not None:
        original_df = st.session_state.df

        with st.form("prediction_form"):
            st.subheader("Enter Post Details")
            input_values = {}
            cols = st.columns(2)

            for i, feature in enumerate(st.session_state.features):
                current_col = cols[i % 2]

                if feature in st.session_state.encoders:
                    unique_values = original_df[feature].dropna().unique()
                    input_values[feature] = current_col.selectbox(
                        f"{feature.replace('_', ' ').title()}", options=unique_values
                    )
                elif original_df[feature].dtype in ['int64', 'float64']:
                    min_val = int(original_df[feature].min())
                    max_val = int(original_df[feature].max())
                    input_values[feature] = current_col.number_input(
                        f"{feature.replace('_', ' ').title()}",
                        min_value=min_val, max_value=max_val, value=min_val
                    )

            predict_button = st.form_submit_button("Predict Virality")

        if predict_button:
            try:
                input_data = []
                for feature in st.session_state.features:
                    if feature in st.session_state.encoders:
                        encoder = st.session_state.encoders[feature]
                        input_data.append(encoder.transform([input_values[feature]])[0])
                    else:
                        input_data.append(input_values[feature])

                prediction = st.session_state.model.predict([input_data])[0]
                probability = st.session_state.model.predict_proba([input_data])[0][1]

                st.subheader("Prediction Result")
                if prediction == 1:
                    st.success(f"🚀 This post is likely to go VIRAL! (Probability: {probability:.2%})")
                else:
                    st.warning(f"📉 This post may not go viral (Probability: {probability:.2%})")

                st.subheader("Key Factors")
                importances = st.session_state.model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': st.session_state.features,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)

                st.bar_chart(importance_df.set_index('Feature').head(5))

            except Exception as e:
                st.error(f"Error making prediction: {e}")
    elif st.session_state.df is None:
        st.warning("Please upload a CSV file to enable predictions")
    elif st.session_state.model is None:
        st.error("Model training failed. Please check your dataset and try again.")
