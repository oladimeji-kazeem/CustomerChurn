import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import io

# Load customer data with updated caching
@st.cache_data
def load_data():
    data = pd.read_csv('data/customer_churn_indiv.csv')  # Replace with the correct path to your data
    return data

# Load dataset
customer_data = load_data()

st.title('Customer Segmentation App')

# Sidebar options
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select an Option', ['Customer Segmentation', 'High-Risk Customers', 'Churn Prediction', 'Churn Drivers', 'Recommendations'])

# Perform Segmentation
def customer_segmentation(data):
    features = data[['Age', 'Premium', 'NumClaims', 'ClaimAmount', 'ClaimSatisfaction']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    data['Segment'] = kmeans.fit_predict(scaled_features)

    # PCA for visualization
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(scaled_features)
    data['PCA1'] = pca_components[:, 0]
    data['PCA2'] = pca_components[:, 1]

    return data

# Segment Insights function
def segment_insights(data):
    insights = {
        0: "Cluster 0: Younger customers with moderate premiums, fewer claims, and moderate satisfaction.",
        1: "Cluster 1: Older customers with higher premiums, more claims, and low claim satisfaction.",
        2: "Cluster 2: Middle-aged customers with low premiums and a low number of claims.",
        3: "Cluster 3: High-premium customers with high claim satisfaction and low churn risk."
    }
    return insights

# Function to export the segment analysis as CSV
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Churn Prediction and Churn Drivers Analysis
def churn_prediction_and_drivers(data):
    # Prepare data for churn prediction
    data_encoded = pd.get_dummies(data, columns=['PaymentHistory', 'PaymentMethod', 'Region', 'State', 'MaritalStatus', 'PolicyType', 'Product', 'Gender'])
    X = data_encoded[['Age', 'Premium', 'NumClaims', 'ClaimAmount', 'ClaimSatisfaction']]
    y = data['ChurnStatus'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Logistic Regression model
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    # Predictions and report
    y_pred = log_reg.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'], output_dict=True)

    # Coefficients (Feature Importance) from Logistic Regression
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': np.abs(log_reg.coef_[0])
    }).sort_values(by='Importance', ascending=False)

    return report, feature_importance

# Segment Visualization and Traits Explanation
if options == 'Customer Segmentation':
    st.write('#### Customer Segmentation')

    # Run customer segmentation
    customer_data = customer_segmentation(customer_data)
    
    st.write('Customer Segments:')
    st.write(customer_data[['CustomerID', 'Segment']].head())

    # Visualize Segments
    fig, ax = plt.subplots()
    scatter = ax.scatter(customer_data['PCA1'], customer_data['PCA2'], c=customer_data['Segment'], cmap='viridis')
    legend1 = ax.legend(*scatter.legend_elements(), title="Segments")
    ax.add_artist(legend1)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    st.pyplot(fig)

    # Ensure only numeric columns are used for aggregation
    numeric_columns = ['Age', 'Premium', 'NumClaims', 'ClaimAmount', 'ClaimSatisfaction']
    segment_analysis = customer_data[numeric_columns + ['Segment']].groupby('Segment').mean()

    st.write('Traits of each Segment:')
    st.dataframe(segment_analysis)

    # Export segment insights as CSV
    csv_data = convert_df_to_csv(segment_analysis)
    st.download_button(
        label="Download Segment Insights as CSV",
        data=csv_data,
        file_name='segment_insights.csv',
        mime='text/csv',
    )

    # Insights for each segment
    st.write('Insights for each Segment:')
    insights = segment_insights(customer_data)
    for segment, insight in insights.items():
        st.write(f"**Segment {segment}:** {insight}")

    # Visualizations for Segment Insights
    st.write('Segment Insights Visualizations:')
    
    # Bar chart for average age per segment
    st.write('Average Age per Segment:')
    fig, ax = plt.subplots()
    ax.bar(segment_analysis.index, segment_analysis['Age'], color='skyblue')
    ax.set_xlabel('Segment')
    ax.set_ylabel('Average Age')
    st.pyplot(fig)

    # Bar chart for average premium per segment
    st.write('Average Premium per Segment:')
    fig, ax = plt.subplots()
    ax.bar(segment_analysis.index, segment_analysis['Premium'], color='orange')
    ax.set_xlabel('Segment')
    ax.set_ylabel('Average Premium')
    st.pyplot(fig)

    # Bar chart for average number of claims per segment
    st.write('Average Number of Claims per Segment:')
    fig, ax = plt.subplots()
    ax.bar(segment_analysis.index, segment_analysis['NumClaims'], color='green')
    ax.set_xlabel('Segment')
    ax.set_ylabel('Average Number of Claims')
    st.pyplot(fig)

    # Bar chart for average claim satisfaction per segment
    st.write('Average Claim Satisfaction per Segment:')
    fig, ax = plt.subplots()
    ax.bar(segment_analysis.index, segment_analysis['ClaimSatisfaction'], color='purple')
    ax.set_xlabel('Segment')
    ax.set_ylabel('Average Claim Satisfaction')
    st.pyplot(fig)

# Identify High-Risk Customers
if options == 'High-Risk Customers':
    st.header('High-Risk Customers Identification')

    # Define risk score based on claims, claim amount, and satisfaction
    customer_data['RiskScore'] = (customer_data['NumClaims'] + customer_data['ClaimAmount'] / customer_data['ClaimAmount'].max() - customer_data['ClaimSatisfaction']) / 3
    high_risk_customers = customer_data[customer_data['RiskScore'] > customer_data['RiskScore'].quantile(0.75)]

    st.write('High-Risk Customers:')
    st.dataframe(high_risk_customers[['CustomerID', 'Age', 'NumClaims', 'ClaimAmount', 'ClaimSatisfaction', 'RiskScore']])

# Churn Prediction
if options == 'Churn Prediction':
    st.header('Churn Prediction')

    report, _ = churn_prediction_and_drivers(customer_data)
    st.write('Churn Prediction Report:')
    st.write(pd.DataFrame(report).transpose())

# Churn Drivers
if options == 'Churn Drivers':
    st.header('Churn Drivers')

    # Get churn drivers
    _, feature_importance = churn_prediction_and_drivers(customer_data)

    st.write('Churn Drivers (Ranked by Feature Importance):')
    st.dataframe(feature_importance)

    # Visualize churn drivers
    st.write('Churn Drivers Visualization:')
    fig, ax = plt.subplots()
    ax.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Features')
    ax.set_title('Churn Drivers (Feature Importance)')
    st.pyplot(fig)

# Recommendations to Reduce Churn
if options == 'Recommendations':
    st.header('Recommendations for Reducing Churn')

    recommendations = """
    ### To reduce customer churn, consider the following strategies:

    1. **Improve Claim Satisfaction**:
        - Monitor claim satisfaction scores and improve the claims process.
    
    2. **Target High-Risk Customers**:
        - Use predictive models to engage with customers likely to churn.
    
    3. **Loyalty Programs and Discounts**:
        - Offer discounts, loyalty programs, and other incentives to long-term customers.
    
    4. **Simplify Payment Processes**:
        - Provide flexible payment options and reminders to avoid missed payments.

    5. **Strengthen Customer Relationships**:
        - Regularly engage with customers and offer personalized support.

    6. **Offer Customizable Policies**:
        - Provide flexible and modular insurance products that adapt to customer needs.
    """
    st.markdown(recommendations)
