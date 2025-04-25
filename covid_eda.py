import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt  # Try moving this after numpy and pandas
import matplotlib as plt  # Try moving this after numpy and pandas
import streamlit as st
import seaborn as sns
from datetime import datetime, timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# Set page config
st.set_page_config(page_title="COVID Data EDA", layout="wide")

# Load data with proper data types
@st.cache_data
def load_data():
    try:
        date_cols = [
            'fecha_reporte_web', 'Fecha_de_notificacion', 'Fecha_de_inicio_de_sintomas',
            'Fecha_de_muerte', 'Fecha_de_recuperacion', 'Fecha_de_diagnostico'
        ]
        
        data = pd.read_csv('CasosCovidAtlantico.csv', parse_dates=date_cols)
        
        # Convert specific columns to appropriate types
        data['ID_de_caso'] = data['ID_de_caso'].astype('int64')
        data['Edad'] = pd.to_numeric(data['Edad'], errors='coerce')
        
        return data
    except FileNotFoundError:
        st.error("File not found. Please make sure 'CasosCovidAtlantico.csv' exists.")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Modified function to prepare time series data with consistent date range
def prepare_time_series(data, date_col, start_date, end_date):
    # Create complete date range from start_date to end_date
    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Count occurrences for each date in the specified column
    ts = data[date_col].value_counts().sort_index()
    
    # Reindex to the full date range, filling missing dates with 0
    ts = ts.reindex(full_date_range, fill_value=0)
    
    return ts


# Function for Holt-Winters forecasting
def holt_winters_forecast(series, forecast_days, test_size=0.2):
    # Split into train and test
    split_idx = int(len(series) * (1 - test_size))
    train, test = series[:split_idx], series[split_idx:]
    
    # Fit model
    #model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=7)
    model = ExponentialSmoothing(series, seasonal='add', seasonal_periods=7)
    model_fit = model.fit()
    
    # Forecast
    forecast = model_fit.forecast(forecast_days)
    
    return train, test, forecast

data = load_data()

if data is not None:
    # Sidebar filters
    st.sidebar.header("Filtros de exploración")

    # Departamento filter
    departamentos = sorted(data['Nombre_departamento'].dropna().unique())
    selected_depto = st.sidebar.selectbox('Departamento', ['Todos'] + departamentos)

    # Municipio filter (dependent on Departamento)
    if selected_depto != 'Todos':
        municipios = sorted(data[data['Nombre_departamento'] == selected_depto]['Nombre_municipio'].dropna().unique())
    else:
        municipios = sorted(data['Nombre_municipio'].dropna().unique())
    selected_municipio = st.sidebar.selectbox('Municipio', ['Todos'] + municipios)

    # Date range filter - using Fecha_de_notificacion as reference
    min_date = data['Fecha_de_notificacion'].min().date()
    max_date = data['Fecha_de_notificacion'].max().date()
    fecha_inicial = st.sidebar.date_input('Fecha inicial', min_date, min_value=min_date, max_value=max_date)
    fecha_final = st.sidebar.date_input('Fecha final', max_date, min_value=min_date, max_value=max_date)

    # Variable selection - exclude specified columns and include date fields
    exclude_cols = ['ID_de_caso', 'Unidad_de_medida_de_edad']
    available_cols = [col for col in data.columns if col not in exclude_cols]
    
    # Group variables by type for better selection
    date_cols = [col for col in available_cols if pd.api.types.is_datetime64_any_dtype(data[col])]
    numeric_cols = [col for col in available_cols if pd.api.types.is_numeric_dtype(data[col]) and col not in date_cols]
    categorical_cols = [col for col in available_cols if col not in numeric_cols and col not in date_cols]
    
    selected_var = st.sidebar.selectbox(
        'Variable a analizar',
        options=categorical_cols + numeric_cols + date_cols,
        format_func=lambda x: f"{x} {'(Fecha)' if x in date_cols else ''}"
    )

    # Apply filters
    filtered_data = data.copy()
    
    # Filter by Departamento
    if selected_depto != 'Todos':
        filtered_data = filtered_data[filtered_data['Nombre_departamento'] == selected_depto]
    
    # Filter by Municipio
    if selected_municipio != 'Todos':
        filtered_data = filtered_data[filtered_data['Nombre_municipio'] == selected_municipio]
    
    # Filter by date range (using Fecha_de_notificacion as reference)
    filtered_data = filtered_data[
        (filtered_data['Fecha_de_notificacion'].dt.date >= fecha_inicial) & 
        (filtered_data['Fecha_de_notificacion'].dt.date <= fecha_final)
    ]

    # Main content
    st.title("Análisis Exploratorio de Datos COVID-19")
    st.write(f"Mostrando datos para: Departamento={selected_depto}, Municipio={selected_municipio}, "
             f"Fecha inicial={fecha_inicial}, Fecha final={fecha_final}")

    # Show filtered data info
    st.subheader("Resumen de datos filtrados")
    st.write(f"Número de registros: {len(filtered_data):,}")
    st.write(filtered_data.head())

    # EDA based on selected variable type
    st.subheader(f"Análisis de: {selected_var}")

    if selected_var in numeric_cols:
        # Numeric variable analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Estadísticas descriptivas**")
            st.write(filtered_data[selected_var].describe())
            
            st.write("**Distribución**")
            fig, ax = plt.subplots()
            sns.histplot(filtered_data[selected_var].dropna(), kde=True, ax=ax)
            st.pyplot(fig)
        
        with col2:
            st.write("**Boxplot**")
            fig, ax = plt.subplots()
            sns.boxplot(y=filtered_data[selected_var].dropna(), ax=ax)
            st.pyplot(fig)
            
            st.write("**Tendencia temporal**")
            if not filtered_data.empty:
                temp_df = filtered_data.groupby(filtered_data['Fecha_de_notificacion'].dt.to_period('M'))[selected_var].mean().reset_index()
                temp_df['Fecha_de_notificacion'] = temp_df['Fecha_de_notificacion'].dt.to_timestamp()
                fig, ax = plt.subplots()
                sns.lineplot(data=temp_df, x='Fecha_de_notificacion', y=selected_var, ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)
    
    elif selected_var in categorical_cols:
        # Categorical variable analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Conteo de categorías**")
            value_counts = filtered_data[selected_var].value_counts(dropna=False)
            st.write(value_counts)
            
            st.write("**Proporciones**")
            st.write(filtered_data[selected_var].value_counts(normalize=True, dropna=False))
        
        with col2:
            st.write("**Gráfico de barras**")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(y=selected_var, data=filtered_data, order=value_counts.index, ax=ax)
            st.pyplot(fig)
        
        # Cross analysis with Departamento if not already selected
        if selected_var != 'Nombre_departamento' and 'Nombre_departamento' in filtered_data.columns:
            st.write("**Distribución por Departamento**")
            pivot_table = pd.crosstab(filtered_data['Nombre_departamento'], filtered_data[selected_var], normalize='index')
            st.write(pivot_table)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            pivot_table.plot(kind='bar', stacked=True, ax=ax)
            plt.xticks(rotation=45)
            plt.legend(title=selected_var, bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)
    
    elif selected_var in date_cols:
        # Date variable analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Rango de fechas**")
            st.write(f"Fecha mínima: {filtered_data[selected_var].min()}")
            st.write(f"Fecha máxima: {filtered_data[selected_var].max()}")
            
            st.write("**Conteo por año**")
            year_counts = filtered_data[selected_var].dt.year.value_counts().sort_index()
            st.write(year_counts)
            
            fig, ax = plt.subplots()
            year_counts.plot(kind='bar', ax=ax)
            plt.title(f"Conteo por año - {selected_var}")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            st.write("**Conteo por mes**")
            month_counts = filtered_data[selected_var].dt.to_period('M').value_counts().sort_index()
            month_counts.index = month_counts.index.astype(str)
            st.write(month_counts)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            month_counts.plot(kind='line', ax=ax)
            plt.title(f"Tendencia mensual - {selected_var}")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            st.write("**Día de la semana**")
            weekday_counts = filtered_data[selected_var].dt.day_name().value_counts()
            st.write(weekday_counts)
            
# Time Series Forecasting Section
    st.header("Pronóstico de Series Temporales")

    # Get number of days to forecast
    forecast_days = st.number_input(
        "Número de días a pronosticar:",
        min_value=1,
        max_value=90,
        value=7,
        step=1
    )

    # In the main code, replace the time series preparation with:
    # Prepare time series data with consistent date range
    start_date = datetime.combine(fecha_inicial, datetime.min.time())
    end_date = datetime.combine(fecha_final, datetime.min.time())

    ts_infectados = prepare_time_series(filtered_data, 'Fecha_de_inicio_de_sintomas', start_date, end_date)
    ts_fallecidos = prepare_time_series(filtered_data, 'Fecha_de_muerte', start_date, end_date)
    ts_recuperados = prepare_time_series(filtered_data, 'Fecha_de_recuperacion', start_date, end_date)
    
    
    # Forecasting for each time series
    if len(ts_infectados) > 0:
        # Infectados forecasting
        st.subheader("Infectados (Fecha_de_inicio_de_sintomas)")
        train_inf, test_inf, forecast_inf = holt_winters_forecast(ts_infectados, forecast_days)
        
        # Create proper date range for forecast that starts right after test period
        forecast_dates = pd.date_range(
            start=test_inf.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        forecast_series = pd.Series(forecast_inf, index=forecast_dates)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        train_inf.plot(label='Entrenamiento', ax=ax, color='blue')
        test_inf.plot(label='Prueba', ax=ax, color='orange')
        forecast_series.plot(label=f'Pronóstico ({forecast_days} días)', ax=ax, color='green')
        
        plt.title(f"Pronóstico de Infectados - {forecast_days} días")
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
        
        # Fallecidos forecasting
        st.subheader("Fallecidos (Fecha_de_muerte)")
        train_fall, test_fall, forecast_fall = holt_winters_forecast(ts_fallecidos, forecast_days)
        
        forecast_dates = pd.date_range(
            start=test_fall.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        #forecast_series = pd.Series(forecast_fall, index=forecast_dates)
        forecast_series = pd.Series(forecast_fall, index=forecast_dates)

        fig, ax = plt.subplots(figsize=(12, 6))
        train_fall.plot(label='Entrenamiento', ax=ax, color='blue')
        test_fall.plot(label='Prueba', ax=ax, color='orange')
        #forecast_series.plot(label=f'Pronóstico ({forecast_days} días)', ax=ax, color='green')
        forecast_fall.plot(label=f'Pronóstico ({forecast_days} días)', ax=ax, color='green')
        
        plt.title(f"Pronóstico de Fallecidos - {forecast_days} días")
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
        
        # Recuperados forecasting
        st.subheader("Recuperados (Fecha_de_recuperacion)")
        train_rec, test_rec, forecast_rec = holt_winters_forecast(ts_recuperados, forecast_days)
        
        forecast_dates = pd.date_range(
            start=test_rec.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        forecast_series = pd.Series(forecast_rec, index=forecast_dates)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        train_rec.plot(label='Entrenamiento', ax=ax, color='blue')
        test_rec.plot(label='Prueba', ax=ax, color='orange')
        forecast_series.plot(label=f'Pronóstico ({forecast_days} días)', ax=ax, color='green')
        
        plt.title(f"Pronóstico de Recuperados - {forecast_days} días")
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
    else:
        st.warning("No hay suficientes datos para realizar pronósticos con los filtros actuales.")
            
    # Show raw data option
    if st.checkbox("Mostrar datos completos filtrados"):
        st.write(filtered_data)

    # Download filtered data
    csv = filtered_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar datos filtrados como CSV",
        data=csv,
        file_name='datos_filtrados.csv',
        mime='text/csv',
    )
else:
    st.warning("No se pudo cargar los datos. Por favor verifique el archivo CSV.")