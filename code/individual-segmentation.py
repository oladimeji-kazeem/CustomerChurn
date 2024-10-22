import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load customer data template for CSV download
def load_template():
    template_data = pd.DataFrame({
        'Age': [],
        'Premium': [],
        'NumClaims': [],
        'ClaimAmount': [],
        'ClaimSatisfaction': []
    })
    return template_data

# Function to download the CSV template
def download_template(template_data):
    csv = template_data.to_csv(index=False).encode('utf-8')
    return csv

# Load customer data for batch upload processing
@st.cache_data
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data

# Train KMeans model on the entire dataset for future predictions
def train_kmeans_model(data):
    features = data[['Age', 'Premium', 'NumClaims', 'ClaimAmount', 'ClaimSatisfaction']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Train KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(scaled_features)

    # Store the scaler and model
    return scaler, kmeans

# Perform Segmentation for batch upload
def perform_segmentation(data, scaler, kmeans):
    features = data[['Age', 'Premium', 'NumClaims', 'ClaimAmount', 'ClaimSatisfaction']]
    scaled_features = scaler.transform(features)

    # Apply KMeans clustering
    data['Segment'] = kmeans.predict(scaled_features)

    # Apply PCA for visualization only if there are enough samples
    if scaled_features.shape[0] > 1:
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(scaled_features)
        data['PCA1'] = pca_components[:, 0]
        data['PCA2'] = pca_components[:, 1]
    else:
        # For a single sample, assign default PCA values
        data['PCA1'] = 0
        data['PCA2'] = 0

    return data

# Provide insights for each segment
def segment_details(data):
    segment_analysis = data.groupby('Segment').mean()[['Age', 'Premium', 'NumClaims', 'ClaimAmount', 'ClaimSatisfaction']]
    st.write("Details of Each Segment:")
    st.dataframe(segment_analysis)

    # Additional insights for each segment
    segment_descriptions = {
        0: "Segment 0: Younger customers with moderate premiums, fewer claims, and high satisfaction.",
        1: "Segment 1: Older customers with higher premiums, more claims, and moderate satisfaction.",
        2: "Segment 2: Middle-aged customers with low premiums, fewer claims, and very high satisfaction.",
        3: "Segment 3: High-premium customers with moderate satisfaction and a high number of claims."
    }

    for segment, description in segment_descriptions.items():
        st.write(f"**Segment {segment}**: {description}")

# Main App Functionality
st.title('Customer Segmentation App')

# Sidebar options
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select an Option', ['Segment an Individual', 'Batch Upload Segmentation', 'Download CSV Template'])

# Load the full dataset to train KMeans once
# Replace this with your actual full dataset
full_dataset = pd.read_csv('data/customer_churn_indiv.csv')  # Use your actual dataset

# Train KMeans on the full dataset
scaler, kmeans = train_kmeans_model(full_dataset)

# Option 1: Segment an individual customer using a form
if options == 'Segment an Individual':
    st.header('Customer Segmentation Form')

    # Create a form for customer input
    with st.form(key='customer_form'):
        age = st.number_input('Age', min_value=0, max_value=120, value=30)
        premium = st.number_input('Premium', min_value=0.0, value=10000.0)
        num_claims = st.number_input('Number of Claims', min_value=0, max_value=100, value=1)
        claim_amount = st.number_input('Total Claim Amount', min_value=0.0, value=5000.0)
        claim_satisfaction = st.slider('Claim Satisfaction (1-5)', 1, 5, value=3)
        
        # Submit form
        submit_button = st.form_submit_button(label='Check Customer Segment')

    if submit_button:
        # Prepare the input data for segmentation
        customer_data = pd.DataFrame({
            'Age': [age],
            'Premium': [premium],
            'NumClaims': [num_claims],
            'ClaimAmount': [claim_amount],
            'ClaimSatisfaction': [claim_satisfaction]
        })

        # Perform segmentation using the pre-trained scaler and model
        segmented_data = perform_segmentation(customer_data, scaler, kmeans)
        segment = segmented_data['Segment'].iloc[0]

        st.write(f"The customer belongs to **Segment {segment}**")

        # Display the segmented data and segment details
        st.write("Segmentation details:")
        st.dataframe(segmented_data)

        # Provide segment details
        segment_details(segmented_data)

# Option 2: Batch Upload for Segmentation
elif options == 'Batch Upload Segmentation':
    st.header('Batch Upload for Customer Segmentation')

    # File uploader
    uploaded_file = st.file_uploader('Upload CSV File', type='csv')
    
    if uploaded_file:
        # Load and segment the data
        customer_data = load_data(uploaded_file)
        
        # Check if necessary columns exist
        required_columns = ['Age', 'Premium', 'NumClaims', 'ClaimAmount', 'ClaimSatisfaction']
        if all(col in customer_data.columns for col in required_columns):
            segmented_data = perform_segmentation(customer_data, scaler, kmeans)

            st.write("Batch Segmentation Results:")
            st.dataframe(segmented_data[['Age', 'Premium', 'NumClaims', 'ClaimAmount', 'ClaimSatisfaction', 'Segment']])

            # Provide segment details
            segment_details(segmented_data)

            # Option to download the segmented results
            segmented_csv = segmented_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Segmentation Results",
                data=segmented_csv,
                file_name='segmented_customers.csv',
                mime='text/csv'
            )

            # Visualize if there are enough samples
            if segmented_data.shape[0] > 1 and 'PCA1' in segmented_data.columns:
                st.write("PCA Visualization of Segments:")
                fig, ax = plt.subplots()
                scatter = ax.scatter(segmented_data['PCA1'], segmented_data['PCA2'], c=segmented_data['Segment'], cmap='viridis')
                legend1 = ax.legend(*scatter.legend_elements(), title="Segments")
                ax.add_artist(legend1)
                ax.set_xlabel('PCA Component 1')
                ax.set_ylabel('PCA Component 2')
                st.pyplot(fig)
            else:
                st.write("Not enough data points for PCA visualization.")
        else:
            st.error('CSV file must contain the following columns: Age, Premium, NumClaims, ClaimAmount, ClaimSatisfaction.')

# Option 3: Download CSV Template
elif options == 'Download CSV Template':
    st.header('Download CSV Template')

    # Provide the template for download
    template_data = load_template()
    csv_data = download_template(template_data)

    st.download_button(
        label="Download Template CSV",
        data=csv_data,
        file_name='customer_template.csv',
        mime='text/csv'
    )

    st.write("The template requires the following columns:")
    st.dataframe(template_data)
