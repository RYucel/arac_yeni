import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re
from fuzzywuzzy import process # Included in case you decide to use fuzzy matching later
import numpy as np # Included in case you decide to use fuzzy matching later

st.title('TRNC New Vehicle Sales Time Series Analysis (Cumulative Sales)')

# Load and Combine Data
@st.cache_data
def load_data():
    file_paths = [
        'Yeni_Arac_2020.csv',
        'Yeni_Arac_2021.csv',
        'Yeni_Arac_2022.csv',
        'Yeni_Arac_2023.csv',
        'Yeni_Arac_2024.csv'
    ]
    dfs = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

combined_df = load_data()

# Data Cleaning and Preparation
combined_df_cleaned = combined_df.fillna(0)
month_columns = [col for col in combined_df_cleaned.columns if '-' in col]
id_vars = ['Aractip', 'Marka', 'Model', 'Arac Kayit Tip']
melted_df = pd.melt(combined_df_cleaned, id_vars=id_vars, value_vars=month_columns, var_name='YearMonth', value_name='Sales')

# Filter for 'Yeni' vehicles
melted_df = melted_df[melted_df['Arac Kayit Tip'] == 'Yeni']

# --- Filter by Aractip ---
allowed_aractips = ['SALON ARAÇ', 'ESTATE']
melted_df = melted_df[melted_df['Aractip'].isin(allowed_aractips)]
# --------------------------

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
        # For example, removing specific words like "series", "edition", etc.
    return model_name

melted_df['Standardized_Model'] = melted_df['Model'].apply(standardize_model_name)
# ---------------------------------

melted_df['YearMonth'] = pd.to_datetime(melted_df['YearMonth'], format='%b-%y')
melted_df['Year'] = melted_df['YearMonth'].dt.year
melted_df['Month'] = melted_df['YearMonth'].dt.month

# Aggregate Sales Data
aggregated_sales = melted_df.groupby(['Standardized_Marka', 'Standardized_Model', 'YearMonth'])['Sales'].sum().reset_index()

# Sort by Standardized_Marka, Standardized_Model, and YearMonth to ensure correct cumulative sum calculation
aggregated_sales = aggregated_sales.sort_values(by=['Standardized_Marka', 'Standardized_Model', 'YearMonth'])

# Calculate cumulative sales for each standardized model within each standardized brand
aggregated_sales['Cumulative_Sales'] = aggregated_sales.groupby(['Standardized_Marka', 'Standardized_Model'])['Sales'].cumsum()


# Create Interactive Visualization
st.sidebar.header('Filter Options')
selected_brand = st.sidebar.selectbox('Select Brand', ['All'] + sorted(aggregated_sales['Standardized_Marka'].unique()))

# Filter models based on selected brand
if selected_brand == 'All':
    filtered_models = sorted(aggregated_sales['Standardized_Model'].unique())
else:
    filtered_models = sorted(aggregated_sales[aggregated_sales['Standardized_Marka'] == selected_brand]['Standardized_Model'].unique())

selected_model = st.sidebar.selectbox('Select Model', ['All'] + filtered_models)

# Filter data for plotting
if selected_brand == 'All':
    if selected_model == 'All':
        plot_data = aggregated_sales
        title = 'Cumulative Sales Trend by Standardized Brand and Model (All)'
    else:
        plot_data = aggregated_sales[aggregated_sales['Standardized_Model'] == selected_model]
        title = f'Cumulative Sales Trend for {selected_model}'
else:
    brand_data = aggregated_sales[aggregated_sales['Standardized_Marka'] == selected_brand]
    if selected_model == 'All':
        plot_data = brand_data
        title = f'Cumulative Sales Trend for {selected_brand} (All Models)'
    else:
        plot_data = brand_data[brand_data['Standardized_Model'] == selected_model]
        title = f'Cumulative Sales Trend for {selected_brand} - {selected_model}'

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
         st.warning("Selecting 'All Brands' and 'All Models' might display too many lines. Please filter by Brand or Model for a clearer view.")
         # Optionally, you could aggregate further or display a different plot type here
         pass # No traces added for this potentially overwhelming case
else:
    st.info("No data available for the selected filters.")


fig.update_layout(
    title=title,
    xaxis_title='Date',
    yaxis_title='Cumulative Number of Sales',
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)