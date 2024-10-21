import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt
from io import BytesIO

# Load the dataset (replace this with the actual dataset file path)
data = pd.read_csv('data/customer_churn_indiv.csv')

# Set page config for better layout and responsiveness
st.set_page_config(layout="wide")

# Logo Display: Put this before the filters to show the logo at the top
st.sidebar.image('image/logo.png', width=100)  # Logo displayed at the top of the sidebar

# Sidebar filters
st.sidebar.header("Filter by Categories")

# Filter by Policy Start Date
policy_start_date = st.sidebar.date_input("Filter Policy Start Date", [])

# Filter by Policy End Date
policy_end_date = st.sidebar.date_input("Filter Policy End Date", [])

# Filter by Gender
gender = st.sidebar.multiselect("Filter Gender", options=data['Gender'].unique(), default=data['Gender'].unique())

# Filter by Marital Status
marital_status = st.sidebar.multiselect("Filter Marital Status", options=data['MaritalStatus'].unique(), default=data['MaritalStatus'].unique())

# Filter by Policy Type
policy_type = st.sidebar.multiselect("Filter Policy Type", options=data['PolicyType'].unique(), default=data['PolicyType'].unique())


# Filter by Payment History
payment_history = st.sidebar.multiselect("Filter Payment History", options=data['PaymentHistory'].unique(), default=data['PaymentHistory'].unique())

# Filter by Payment Methods
payment_method = st.sidebar.multiselect("Filter Payment Method", options=data['PaymentMethod'].unique(), default=data['PaymentMethod'].unique())

# Apply filters
filtered_data = data[
    (data['Gender'].isin(gender)) &
    (data['MaritalStatus'].isin(marital_status)) &
    (data['PolicyType'].isin(policy_type)) &
    (data['PaymentHistory'].isin(payment_history)) &
    (data['PaymentMethod'].isin(payment_method))
]

# Date filter application
if policy_start_date:
    filtered_data = filtered_data[filtered_data['PolicyStartDate'] >= pd.to_datetime(policy_start_date)]
if policy_end_date:
    filtered_data = filtered_data[filtered_data['PolicyEndDate'] <= pd.to_datetime(policy_end_date)]

# Function to load CSV or Excel file
def load_data(file):
    if file.name.endswith('.csv'):
        data = pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        data = pd.read_excel(file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None
    return data

# Template download links
@st.cache_data
def create_template_csv():
    df_template = pd.DataFrame({
        'CustomerID': ['CS000001', 'CS000002'],
        'Age': [35, 42],
        'Gender': ['Male', 'Female'],
        'Location': ['Lagos', 'Abuja'],
        'MaritalStatus': ['Married', 'Single'],
        'PolicyType': ['Auto', 'Home'],
        'PolicyStartDate': ['2022-01-15', '2023-04-20'],
        'PolicyEndDate': ['2024-01-15', '2025-04-20'],
        'Premium': [150000.00, 250000.00],
        'NumClaims': [1, 0],
        'ClaimAmount': [50000.00, 0],
        'PaymentHistory': ['On-time', 'Late'],
        'PaymentMethod': ['Bank Transfer', 'Mobile Payment'],
        'ChurnStatus': ['No', 'Yes']
    })
    return df_template

# Download template CSV
csv_template = create_template_csv()
csv_template_file = csv_template.to_csv(index=False).encode('utf-8')

# Create Excel template in memory
def create_template_excel(df_template):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_template.to_excel(writer, index=False)
    processed_data = output.getvalue()
    return processed_data

# Streamlit app layout
st.title("Customer Data Analytics")

st.write("#### Download Templates to Fill and Upload")
col1, col2 = st.columns(2)

with col1:
    st.download_button(label="Download CSV Template", data=csv_template_file, file_name="customer_data_template.csv", mime="text/csv")

with col2:
    excel_template_file = create_template_excel(csv_template)
    st.download_button(label="Download Excel Template", data=excel_template_file, file_name="customer_data_template.xlsx", mime="application/vnd.ms-excel")

st.write("#### Upload Your CSV or Excel File")

# File uploader for CSV or Excel
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

# Process the uploaded file
if uploaded_file:
    data = load_data(uploaded_file)
    
    if data is not None:
        st.success("File uploaded successfully!")
        
        # Display the first few rows of the uploaded file
        st.write("### Preview of Uploaded Data")
        st.write(data.head())
        
        # Add your visualizations here, using the 'data' variable
        # Example: Total Premium
        total_premium_million = data['Premium'].sum() / 1_000_000
        total_customers = len(data)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Customers", f"{total_customers:,}")  # With thousand separators
        with col2:
            st.metric("Total Premium (M)", f"₦{total_premium_million:.2f}M")
        
        # You can add your existing visualizations, such as age distribution, gender breakdown, etc.
        # Example: Age Group distribution
        age_bins = [18, 28, 38, 48, 58, 68, 78]
        age_labels = ['18-28', '29-38', '39-48', '49-58', '59-68', '69-78']
        data['AgeGroup'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels, right=False)

        age_distribution = data['AgeGroup'].value_counts().sort_index()
        age_bar_chart = px.bar(age_distribution, x=age_distribution.index, y=age_distribution.values, 
                               labels={'x': 'Age Group', 'y': 'Number of Customers'},
                               title='Age Distribution by Group',
                               color_discrete_sequence=px.colors.qualitative.Set3)

        st.plotly_chart(age_bar_chart, use_container_width=True)
else:
    st.info("Please upload a CSV or Excel file.")

# Metrics Section
st.title("Analytics Dashboard")

# Define columns for metrics
col1, col2, col3, col4, col5 = st.columns(5)
total_customers = len(filtered_data)
total_premium = filtered_data['Premium'].sum()/1_000_000_000
total_claim = filtered_data['ClaimAmount'].sum()/1_000_000_000
total_claims = filtered_data['NumClaims'].sum()
total_products = filtered_data['PolicyType'].nunique()

# Display metrics
with col1:
    st.metric("Total Customers", f"{total_customers:,}")
with col2:
    st.metric("Total Premium", f"₦{total_premium:,.2f}bn")
with col3:
    st.metric("Total Claim Amount", f"₦{total_claim:,.2f}bn")
with col4:
    st.metric("Total Claims", f"{total_claims:,}")
with col5:
    st.metric("Nos of Products", total_products)

# Chart Section
st.subheader("Charts Overview")

# Row 1: Gender Distribution and Age Distribution
col1, col2 = st.columns(2)

# Pie chart - Gender Distribution
with col1:
    pie_chart_gender = px.pie(filtered_data, names='Gender', title='Gender Distribution', color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(pie_chart_gender, use_container_width=True)

# Age Histogram
with col2:
    age_histogram = px.histogram(filtered_data, x='Age', nbins=20, title='Age Distribution', color_discrete_sequence=px.colors.qualitative.Set1)
    age_histogram.update_layout(xaxis_title="Age", yaxis_title="Count")
    st.plotly_chart(age_histogram, use_container_width=True)

# Row 2: Premium by Product and Claims vs Churn Status
col1, col2 = st.columns(2)

# Bar chart - Premium by Product
with col1:
    bar_chart_product = px.bar(filtered_data.groupby('Product').sum().reset_index(), x='Product', y='Premium', title="Premium by Product", labels={"Premium": "Total Premium (₦)", "Product": "Product"})
    st.plotly_chart(bar_chart_product, use_container_width=True)

# Scatter plot - Claims vs Churn Status
with col2:
    scatter_claims_churn = px.scatter(filtered_data, x='NumClaims', y='ClaimAmount', color='ChurnStatus', title="Claims vs Churn Status", labels={"NumClaims": "Number of Claims", "ClaimAmount": "Claim Amount (₦)"})
    st.plotly_chart(scatter_claims_churn, use_container_width=True)

# Row 3: Payment History and Policy Type
col1, col2 = st.columns(2)

# Bar chart - Payment History
with col1:
    bar_chart_payment = px.bar(filtered_data.groupby('PaymentHistory').count().reset_index(), x='PaymentHistory', y='CustomerID', title="Customers by Payment History", labels={"PaymentHistory": "Payment History", "CustomerID": "Number of Customers"})
    st.plotly_chart(bar_chart_payment, use_container_width=True)

# Grouped Bar Chart - Policy Type vs Claims
with col2:
    grouped_bar_policy = px.bar(filtered_data, x="PolicyType", y="NumClaims", color="PolicyType", barmode="group", title="Policy Type vs Number of Claims", labels={"NumClaims": "Number of Claims", "PolicyType": "Policy Type"})
    st.plotly_chart(grouped_bar_policy, use_container_width=True)

# Trends over Time (Annual, Quarterly, Monthly)
st.subheader("Trends Over Time")

# Convert policy dates to datetime
filtered_data['PolicyStartDate'] = pd.to_datetime(filtered_data['PolicyStartDate'], dayfirst=True, errors='coerce')
filtered_data['PolicyEndDate'] = pd.to_datetime(filtered_data['PolicyEndDate'], dayfirst=True, errors='coerce')

# Extract year, quarter, and month for trend analysis
filtered_data['Year'] = filtered_data['PolicyStartDate'].dt.year
filtered_data['Quarter'] = filtered_data['PolicyStartDate'].dt.to_period('Q')
filtered_data['Month'] = filtered_data['PolicyStartDate'].dt.to_period('M')

# Trend: Annual Premium Trend
annual_trend = filtered_data.groupby('Year').agg({'Premium': 'sum'}).reset_index()
# annual_trend = px.line(filtered_data.groupby('Year').sum().reset_index(), x='Year', y='Premium', title="Annual Premium Trend")
#st.plotly_chart(annual_trend, use_container_width=True)
annual_trend = px.line(annual_trend, x='Year', y='Premium', title="Annual Premium Trend", labels={'Premium': 'Total Premium (₦)', 'Year': 'Year'})
st.plotly_chart(annual_trend, use_container_width=True)

# Trend: Quarterly Premium Trend
# Group by 'Quarter' and sum numeric fields (like Premium)
quarterly_trend = filtered_data.groupby(filtered_data['Quarter'].astype(str)).agg({'Premium': 'sum'}).reset_index()

# Plot the quarterly trend
quarterly_trend = px.line(quarterly_trend, x='Quarter', y='Premium', title="Quarterly Premium Trend", labels={'Premium': 'Total Premium (₦)', 'Quarter': 'Quarter'})
st.plotly_chart(quarterly_trend, use_container_width=True)

# Trend: Monthly Premium Trend
# Group by 'Month' and sum numeric fields (like Premium)
monthly_trend = filtered_data.groupby(filtered_data['Month'].astype(str)).agg({'Premium': 'sum'}).reset_index()

# Plot the monthly trend
monthly_trend = px.line(monthly_trend, x='Month', y='Premium', title="Monthly Premium Trend", labels={'Premium': 'Total Premium (₦)', 'Month': 'Month'})
st.plotly_chart(monthly_trend, use_container_width=True)

# Table for Claims and Premium Summary
st.subheader("Claims and Premium Summary")

# Define previous periods
today = dt.date.today()
last_month = (today.replace(day=1) - dt.timedelta(days=1)).replace(day=1)
last_quarter = (today - pd.DateOffset(months=3)).to_period('Q').start_time
last_year = today.replace(year=today.year - 1)

# Filter for the periods
# previous_month_data = filtered_data[filtered_data['PolicyStartDate'].dt.to_period('M') == last_month.to_period('M')]
# Filter for the previous month
filtered_data['PolicyStartMonth'] = filtered_data['PolicyStartDate'].dt.to_period('M')
previous_month_data = filtered_data[filtered_data['PolicyStartMonth'] == pd.Period(last_month, 'M')]

filtered_data['PolicyStartQuarter'] = filtered_data['PolicyStartDate'].dt.to_period('Q')
previous_quarter_data = filtered_data[filtered_data['PolicyStartQuarter'] == pd.Period(last_quarter, 'Q')]

previous_year_data = filtered_data[filtered_data['PolicyStartDate'].dt.year == last_year.year]

# Define summary table
summary_table = {
    'previous_month': [previous_month_data['NumClaims'].sum(), previous_month_data['PolicyType'].count(), previous_month_data['ClaimAmount'].sum(), previous_month_data['Premium'].sum()],
    'previous_quarter': [previous_quarter_data['NumClaims'].sum(), previous_quarter_data['PolicyType'].count(), previous_quarter_data['ClaimAmount'].sum(), previous_quarter_data['Premium'].sum()],
    'previous_year': [previous_year_data['NumClaims'].sum(), previous_year_data['PolicyType'].count(), previous_year_data['ClaimAmount'].sum(), previous_year_data['Premium'].sum()],
    'current_month': [filtered_data['NumClaims'].sum(), filtered_data['PolicyType'].count(), filtered_data['ClaimAmount'].sum(), filtered_data['Premium'].sum()],
}

summary_df = pd.DataFrame(summary_table, index=['Nos of Claims', 'Nos of Policies', 'Claim Amount', 'Premium'])

# Calculate variance
summary_df['monthly_variance'] = summary_df['current_month'] - summary_df['previous_month']
summary_df['quarter_variance'] = summary_df['current_month'] - summary_df['previous_quarter']
summary_df['annual_variance'] = summary_df['current_month'] - summary_df['previous_year']

# Display summary table
st.table(summary_df)

