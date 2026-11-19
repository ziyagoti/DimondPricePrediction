import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


# Regression Models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR

# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

# screen aakha ma avi jase
st.set_page_config(
    page_title="My Wide App",
    layout="wide",  # Options: "centered" (default) or "wide"
    initial_sidebar_state="expanded",
)

st.title("Diamond Price Prediction")

## Data Load
df = pd.read_csv("C:/TOPS/ML/DimondPred/diamonds.csv")

# unnamed columns delete
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

num_col = ['carat', 'depth', 'table', 'x', 'y', 'z']
cat_col = ['cut', 'color', 'clarity']

## Sidebar ma checkbox if click it than show data
b1 = st.sidebar.checkbox("Show Dataset")
if b1:
    st.write(df.head())

## EDA
col1, col2 = st.columns([1, 1])

b2 = st.sidebar.checkbox("EDA")
if b2:
    st.write("EDA")
    s1 = st.sidebar.selectbox("Types of EDA", ['Univariate', 'Bivariate', 'Multivariate'])

    if s1 == 'Univariate':
        col_name = st.sidebar.selectbox("Select Column", df.columns)
        if col_name in num_col or col_name == 'price':
            chart1 = px.histogram(df[col_name])
            col1.plotly_chart(chart1)

            charts2 = px.box(df[col_name])
            col2.plotly_chart(charts2)

        elif col_name in cat_col:
            df_cat_eda = df.groupby(col_name)['price'].count().reset_index()
            df_cat_eda.rename(columns={'price': 'Count'}, inplace=True)
            st.write(df_cat_eda)

            charts1 = px.bar(df_cat_eda, col_name, 'Count')
            col1.plotly_chart(charts1)

            charts2 = px.pie(df_cat_eda, col_name, 'Count')
            col2.plotly_chart(charts2)

    elif s1 == 'Bivariate':
        col_name = st.sidebar.selectbox("Select Feature Column", num_col)
        target_col = df['price']

        chart1 = px.scatter(df, col_name, target_col)
        col1.plotly_chart(chart1)

        df_num_eda = df.groupby(col_name)['price'].mean().reset_index()
        charts2 = px.line(df_num_eda, col_name, 'price')
        col2.plotly_chart(charts2)

    st.write(s1)

# trainTest split
X = df.drop(columns=['price'])
Y = df['price']

# min,max,default,step
test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.9, value=0.2, step=0.1, key="test_size")
random_state = st.sidebar.number_input("Random State", min_value=0, max_value=100, value=3, step=1, key="random_state")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

b3 = st.sidebar.checkbox("Show Train-Test Split Summary")
if b3:
    st.write(f"X Train Shape: {X_train.shape}")
    st.write(f"X Test Shape: {X_test.shape}")
    st.write(f"Y Train Shape: {Y_train.shape}")
    st.write(f"Y Test Shape: {Y_test.shape}")

# Preprocessing
scaler = StandardScaler()
encoder = OrdinalEncoder()

num_train = X_train[num_col]
num_test = X_test[num_col]

num_train_scaled = scaler.fit_transform(num_train)
num_train_scaled_df = pd.DataFrame(num_train_scaled, columns=num_col, index=num_train.index)

num_test_scaled = scaler.transform(num_test)
num_test_scaled_df = pd.DataFrame(num_test_scaled, columns=num_col, index=num_test.index)

cat_train = X_train[cat_col]
cat_test = X_test[cat_col]

cat_train_encoded = encoder.fit_transform(cat_train)
cat_train_encoded_df = pd.DataFrame(cat_train_encoded, columns=cat_col, index=cat_train.index)

cat_test_encoded = encoder.transform(cat_test)
cat_test_encoded_df = pd.DataFrame(cat_test_encoded, columns=cat_col, index=cat_test.index)

x_train_scaled_encoded = num_train_scaled_df.join(cat_train_encoded_df)
x_test_scaled_encoded = num_test_scaled_df.join(cat_test_encoded_df)

# Model Selection Sidebar
st.sidebar.markdown("---")
model_list = ['Decision Tree Regressor', 'Random Forest Regressor', 'Gradient Boosting Regressor',
              'Linear Regression', 'Ridge Regression', 'Lasso Regression', 'K-Nearest Neighbors Regressor',
              'AdaBoost Regressor', 'Support Vector Regressor', 'Polynomial Regression']

model_choice = st.sidebar.selectbox("Select a model for training", model_list)

# Model Object Creation (સુધારેલું)
if model_choice == 'Decision Tree Regressor':
    model = DecisionTreeRegressor()
elif model_choice == 'Random Forest Regressor':
    model = RandomForestRegressor(n_estimators=50)  # ઝડપી રન કરવા માટે 50 સેટ કર્યા છે
elif model_choice == 'Gradient Boosting Regressor':
    model = GradientBoostingRegressor()
elif model_choice == 'Linear Regression':
    model = LinearRegression()
elif model_choice == 'Ridge Regression':
    model = Ridge()
elif model_choice == 'Lasso Regression':
    model = Lasso()
elif model_choice == 'K-Nearest Neighbors Regressor':
    model = KNeighborsRegressor()
elif model_choice == 'AdaBoost Regressor':
    model = AdaBoostRegressor()
elif model_choice == 'Support Vector Regressor':
    model = SVR()
elif model_choice == 'Polynomial Regression':
    degree = st.sidebar.slider("Select the degree for Polynomial Regression", min_value=2, max_value=5, value=2, step=1)
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# Training and Results (સુધારેલું)
b5 = st.checkbox("Show Result")
if b5:
    with st.spinner(f"Training {model_choice}... Please wait..."):
        model.fit(x_train_scaled_encoded, Y_train)

        Y_pred = model.predict(x_test_scaled_encoded)

        mse = mean_squared_error(Y_test, Y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)
        mape = mean_absolute_percentage_error(Y_test, Y_pred)

        st.write(f"MSE:-{mse}")
        st.write(f"RMSE:-{rmse}")
        st.write(f"MAE:-{mae}")
        st.write(f"R2 Score:-{r2}")

        models = {'Pred_model': model, "Scaler": scaler, "Encoder": encoder}
        # Convert dictionary into pickle bytes
        model_bytes = pickle.dumps(models)

        # Download button
        st.download_button(
            label="Download Models",
            data=model_bytes,
            file_name="models.pkl")