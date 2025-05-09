# Vehicle Sales Time Series Analysis
# This code is a Streamlit application that visualizes vehicle sales data over time.
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re
from fuzzywuzzy import process # Included in case you decide to use fuzzy matching later
import numpy as np # Included in case you decide to use fuzzy matching later

st.title('Vehicle Sales Time Series Analysis (New vs. Used Cars)')

# Load and Combine Data
@st.cache_data
def load_data():
    new_car_file_paths = [
        'Yeni_Arac_2020.csv',
        'Yeni_Arac_2021.csv',
        'Yeni_Arac_2022.csv',
        'Yeni_Arac_2023.csv',
        'Yeni_Arac_2024.csv'
    ]
    used_car_file_paths = [
        'Kullanilmis_Arac_2021.csv',
        'Kullanilmis_Arac_2022.csv',
        'Kullanilmis_Arac_2023.csv',
        'Kullanilmis_Arac_2024.csv'
    ]

    new_dfs = []
    for file_path in new_car_file_paths:
        df = pd.read_csv(file_path)
        new_dfs.append(df)
    combined_new_df = pd.concat(new_dfs, ignore_index=True)

    used_dfs = []
    for file_path in used_car_file_paths:
        df = pd.read_csv(file_path)
        used_dfs.append(df)
    combined_used_df = pd.concat(used_dfs, ignore_index=True)

    return combined_new_df, combined_used_df

combined_new_df, combined_used_df = load_data()

# Data Cleaning and Preparation
@st.cache_data
def clean_and_prepare_data(df, vehicle_type):
    df_cleaned = df.fillna(0)
    month_columns = [col for col in df_cleaned.columns if '-' in col]
    id_vars = ['Aractip', 'Marka', 'Model', 'Arac Kayit Tip']
    melted_df = pd.melt(df_cleaned, id_vars=id_vars, value_vars=month_columns, var_name='YearMonth', value_name='Sales')

    # Filter by Arac Kayit Tip
    melted_df = melted_df[melted_df['Arac Kayit Tip'] == vehicle_type]

    # Convert 'Sales' column to numeric
    melted_df['Sales'] = pd.to_numeric(melted_df['Sales'], errors='coerce').fillna(0)


    # --- Brand Name Standardization ---
    def standardize_brand_name(brand_name):
        if isinstance(brand_name, str):
            brand_name = brand_name.upper().strip()
            if 'MERCEDES' in brand_name:
                return 'MERCEDES-BENZ'
            # Add other specific brand standardizations here if needed
        return brand_name

    melted_df['Standardized_Marka'] = melted_df['Marka'].apply(standardize_brand_name)
    # ---------------------------------

    # --- Model Name Standardization ---
    def standardize_model_name(model_name):
        if isinstance(model_name, str):
            # Convert to lowercase
            model_name = model_name.lower()
            # Remove extra spaces and leading/trailing spaces
            model_name = re.sub(r'\s+', ' ', model_name).strip()
            # Remove hyphens and underscores
            model_name = model_name.replace('-', '').replace('_', '')
            # You might want to add more specific cleaning rules here
        return model_name

    melted_df['Standardized_Model'] = melted_df['Model'].apply(standardize_model_name)
    # ---------------------------------


    melted_df['YearMonth'] = pd.to_datetime(melted_df['YearMonth'], format='%b-%y')
    melted_df['Year'] = melted_df['YearMonth'].dt.year
    melted_df['Month'] = melted_df['YearMonth'].dt.month

    return melted_df

# Aggregate Sales Data
@st.cache_data
def aggregate_sales_data(melted_df):
    aggregated_sales = melted_df.groupby(['Standardized_Marka', 'Standardized_Model', 'YearMonth'])['Sales'].sum().reset_index()
    aggregated_sales = aggregated_sales.sort_values(by=['Standardized_Marka', 'Standardized_Model', 'YearMonth'])
    aggregated_sales['Cumulative_Sales'] = aggregated_sales.groupby(['Standardized_Marka', 'Standardized_Model'])['Sales'].cumsum()
    return aggregated_sales


# Streamlit App Layout
st.sidebar.header('Select Vehicle Type')
vehicle_type = st.sidebar.radio('Choose Vehicle Type', ('Yeni', 'Kullanılmış'))

if vehicle_type == 'Yeni':
    st.header('New Vehicle Sales Analysis')
    melted_data = clean_and_prepare_data(combined_new_df, 'Yeni')
    # Add Aractip filter for New Cars if needed
    allowed_aractips_new = ['SALON ARAÇ', 'ESTATE']
    melted_data = melted_data[melted_data['Aractip'].isin(allowed_aractips_new)]

else: # Kullanılmış
    st.header('Used Vehicle Sales Analysis')
    melted_data = clean_and_prepare_data(combined_used_df, 'Kullanılmış')


aggregated_sales = aggregate_sales_data(melted_data)

# Create Interactive Visualization
st.sidebar.header('Filter Options')
# Use standardized brand and model names for dropdowns
brands = sorted(aggregated_sales['Standardized_Marka'].unique())
selected_brand = st.sidebar.selectbox('Select Brand', ['All'] + brands)

# Filter models based on selected brand
if selected_brand == 'All':
    filtered_models = sorted(aggregated_sales['Standardized_Model'].unique())
else:
    filtered_models = sorted(aggregated_sales[aggregated_sales['Standardized_Marka'] == selected_brand]['Standardized_Model'].unique())

selected_model = st.sidebar.selectbox('Select Model', ['All'] + filtered_models)

# Filter data for plotting and table display
if selected_brand == 'All':
    if selected_model == 'All':
        plot_data = aggregated_sales
        title = f'Cumulative Sales Trend by Standardized Brand and Model ({vehicle_type} - All)'
    else:
        plot_data = aggregated_sales[aggregated_sales['Standardized_Model'] == selected_model]
        title = f'Cumulative Sales Trend for {vehicle_type} - {selected_model}'
else:
    brand_data = aggregated_sales[aggregated_sales['Standardized_Marka'] == selected_brand]
    if selected_model == 'All':
        plot_data = brand_data
        title = f'Cumulative Sales Trend for {vehicle_type} - {selected_brand} (All Models)'
    else:
        plot_data = brand_data[brand_data['Standardized_Model'] == selected_model]
        title = f'Cumulative Sales Trend for {vehicle_type} - {selected_brand} - {selected_model}'

fig = go.Figure()

if not plot_data.empty:
    if selected_model == 'All' and selected_brand != 'All':
         for model in plot_data['Standardized_Model'].unique():
             model_data = plot_data[plot_data['Standardized_Model'] == model]
             fig.add_trace(go.Scatter(x=model_data['YearMonth'], y=model_data['Cumulative_Sales'], mode='lines', name=model))
    elif selected_model != 'All':
         fig.add_trace(go.Scatter(x=plot_data['YearMonth'], y=plot_data['Cumulative_Sales'], mode='lines', name=selected_model if selected_brand == 'All' else f"{selected_brand} - {selected_model}"))
    elif selected_brand == 'All' and selected_model == 'All':
         # This case might result in too many lines, consider limiting or changing visualization type
         st.warning(f"Selecting 'All Brands' and 'All Models' for {vehicle_type} vehicles might display too many lines. Please filter by Brand or Model for a clearer view.")
         # Optionally, you could aggregate further or display a different plot type here
         pass # No traces added for this potentially overwhelming case
else:
    st.info(f"No data available for the selected filters for {vehicle_type} vehicles.")


fig.update_layout(
    title=title,
    xaxis_title='Date',
    yaxis_title='Cumulative Number of Sales',
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# --- Display filtered data in a table ---
if not plot_data.empty:
    st.subheader(f'{vehicle_type} Vehicle Sales Data Table')
    # Select relevant columns to display
    table_data = plot_data[['Standardized_Marka', 'Standardized_Model', 'YearMonth', 'Sales', 'Cumulative_Sales']]
    st.dataframe(table_data.set_index('YearMonth'))
else:
    st.info("No data to display in the table.")
# ---------------------------------------