import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Set up the page layout
st.set_page_config(page_title="Data Explorer & Predictor", layout="wide")

# --- Main Content ---
st.title("Interactive Data Explorer & Basic Prediction Model")

# --- Upload the data ---
st.header("Upload Your Data")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    # Load data
    data = pd.read_csv(uploaded_file)

    # Use columns to display Data Preview and Basic Statistics side by side
    if len(data.columns) <= 10:
        if len(data.columns) <= 5:
            empty_col1, col1, col2, empty_col2 = st.columns([0.25, 0.25, 0.25, 0.25])
        else:
            empty_col1, col1, col2, empty_col2 = st.columns([0.1, 0.4, 0.4, 0.1])
    else:
        # Standard layout with 80-20 split
        col1, col2 = st.columns([0.8, 0.2])

    with col1:
        st.subheader("Data Preview")
        st.write(data.head(8))

    with col2:
        st.subheader("Basic Statistics")
        st.write(data.describe())

    # --- Data Visualization ---
    st.header("Data Visualization")
    plot_type = st.selectbox(
        "Choose Plot Type", ["Bar Plot", "Histogram", "Scatter Plot"]
    )

    if plot_type == "Bar Plot":
        categorical_columns = data.select_dtypes(include=["object", "category"]).columns
        x_column = st.selectbox("X-axis (categorical only)", categorical_columns)

        # Limit number of categories for readability
        if data[x_column].dtype == "object":
            # Drop NaN values and count frequency of categories
            data_cleaned = data[x_column].dropna()
            category_counts = data_cleaned.value_counts()

            # Check if the column has more than 2 unique values
            if len(category_counts) > 2:
                # Top N categories
                top_n = st.slider(
                    "Select Top N Categories", min_value=2, max_value=10, value=5
                )

                # Select only the top N categories
                top_categories = category_counts.head(top_n)
            else:
                # If there are only two unique values, no need to apply "Top N"
                top_categories = category_counts

            # Plot the bar chart for top N categories or the full categories
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(x=top_categories.index, y=top_categories.values)
            ax.set_xlabel(x_column)
            ax.set_ylabel("Count")
            st.pyplot(plt)

    elif plot_type == "Histogram":
        numerical_columns = data.select_dtypes(include=["number"]).columns
        x_column = st.selectbox("X-axis (numeric only)", numerical_columns)

        # Adjust number of bins based on the variable's range
        if pd.api.types.is_numeric_dtype(data[x_column]):
            plt.figure(figsize=(12, 6))

            # Slider for number of bins based on the range of the selected column
            bins = st.slider(
                f"Select number of bins for {x_column}",
                2,  # Set a lower limit of 2 for number of bins
                data[x_column].count(),  # Set an upper limit for number of bins
                int(data[x_column].median()),  # Set the default value to the median
            )

            # Ensure at least 2 bins are selected
            if bins < 2:
                bins = 2

            sns.histplot(data[x_column], kde=True, bins=bins)
            st.pyplot(plt)

    elif plot_type == "Scatter Plot":
        numerical_columns = data.select_dtypes(include=["number"]).columns
        x_column = st.selectbox("X-axis", numerical_columns)
        y_column = st.selectbox("Y-axis", numerical_columns)

        if pd.api.types.is_numeric_dtype(
            data[x_column]
        ) and pd.api.types.is_numeric_dtype(data[y_column]):
            plt.figure(figsize=(10, 5))
            sns.scatterplot(data=data, x=x_column, y=y_column)
            st.pyplot(plt)

    # --- Basic Prediction Model ---
    st.header("Basic Prediction Model")
    model_type = st.selectbox(
        "Select Model Type", ["Linear Regression", "Decision Tree", "Random Forest"]
    )

    # Select numeric columns for features and target
    numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()
    features = st.multiselect(
        "Select Feature Columns (Numeric)", options=numeric_columns
    )
    target = st.selectbox("Select Target Column", options=numeric_columns)

    if features and target:
        # Drop rows where target is NaN
        data = data.dropna(subset=[target])

        # Ensure the feature columns are selected
        X = data[features]
        y = data[target]

        # Apply scaling to numerical features
        numeric_features = X.select_dtypes(include=["number"]).columns.tolist()

        # Handle missing values for both numerical and categorical features
        numeric_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="mean"),
                ),
                ("scaler", StandardScaler()),  # Scale the features
            ]
        )

        categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()
        categorical_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="constant", fill_value="missing"),
                ),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore"),
                ),  # One-hot encode categorical variables
            ]
        )

        # Apply transformations to the features
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_columns),
            ],
            remainder="drop",
        )

        # Select model type based on user input
        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Decision Tree":
            model = DecisionTreeRegressor(random_state=42)
        elif model_type == "Random Forest":
            model = RandomForestRegressor(random_state=42)

        # Create a pipeline to chain preprocessing and model fitting
        model_pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", model)]
        )

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Fit the model
        model_pipeline.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model_pipeline.predict(X_test)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Display error metrics
        st.write(f"Model Type: {model_type}")
        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"R-squared (R²): {r2}")

        # Prediction Interface
        st.subheader("Make Predictions")
        user_input = {
            col: st.number_input(
                f"Enter {col} value", float(X[col].min()), float(X[col].max())
            )
            for col in features
        }
        user_input_df = pd.DataFrame([user_input])

        if st.button("Predict"):
            prediction = model_pipeline.predict(user_input_df)
            st.write(f"Predicted {target}: {prediction[0]}")
    else:
        st.warning("Please select feature columns and a target column.")

    # --- Data Filtering ---
    st.header("Data Filtering")
    filter_columns = st.multiselect("Select columns to filter by", data.columns)
    filtered_data = data.copy()

    for column in filter_columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            min_val, max_val = float(data[column].min()), float(data[column].max())
            user_min, user_max = st.slider(
                f"Filter by {column}", min_val, max_val, (min_val, max_val)
            )
            filtered_data = filtered_data[
                (filtered_data[column] >= user_min)
                & (filtered_data[column] <= user_max)
            ]
        else:
            filter_val = st.selectbox(f"Filter by {column}", data[column].unique())
            filtered_data = filtered_data[filtered_data[column] == filter_val]

    st.write("Filtered Data", filtered_data)

    # --- Export Filtered Data ---
    st.header("Export Filtered Data")
    csv_data = filtered_data.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv_data,
        file_name="filtered_data.csv",
        mime="text/csv",
    )
else:
    st.info("Upload a CSV file to proceed.")

# --- About Section ---
st.header("About")
st.info(
    "Quickly upload, visualize, filter, and analyze your data, then build a simple prediction model and export results."
)
