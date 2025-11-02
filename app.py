import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# Configure the app
st.set_page_config(
    page_title="Kenya Climate AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("🌍 Kenya Climate AI Analyzer")
st.markdown("""
**Real-time climate analysis for all 47 Kenyan counties using NASA POWER data**
- Compare temperature, rainfall, and solar radiation trends
- AI-powered risk assessment and recommendations
- Data source: [NASA POWER API](https://power.larc.nasa.gov/)
""")

# All 47 Kenyan counties with verified coordinates
COUNTIES = {
    "Baringo": (0.4667, 35.9667),
    "Bomet": (-0.7833, 35.3417),
    "Bungoma": (0.5667, 34.5667),
    "Busia": (0.4600, 34.1117),
    "Elgeyo-Marakwet": (0.5100, 35.2700),
    "Embu": (-0.5390, 37.4574),
    "Garissa": (-0.4532, 39.6461),
    "Homa Bay": (-0.5360, 34.4500),
    "Isiolo": (0.3556, 37.5833),
    "Kajiado": (-1.8524, 36.7767),
    "Kakamega": (0.2827, 34.7519),
    "Kericho": (-0.3670, 35.2833),
    "Kiambu": (-1.0314, 36.8685),
    "Kilifi": (-3.5107, 39.9093),
    "Kirinyaga": (-0.4990, 37.2803),
    "Kisii": (-0.6833, 34.7667),
    "Kisumu": (-0.0917, 34.7680),
    "Kitui": (-1.3670, 38.0106),
    "Kwale": (-4.1816, 39.4606),
    "Laikipia": (0.2041, 36.5580),
    "Lamu": (-2.2696, 40.9000),
    "Machakos": (-1.5177, 37.2634),
    "Makueni": (-1.8000, 37.6200),
    "Mandera": (3.9365, 41.8675),
    "Marsabit": (2.3340, 37.9900),
    "Meru": (0.0515, 37.6456),
    "Migori": (-1.0667, 34.4667),
    "Mombasa": (-4.0435, 39.6682),
    "Murang'a": (-0.7833, 37.0333),
    "Nairobi": (-1.2864, 36.8172),
    "Nakuru": (-0.3031, 36.0800),
    "Nandi": (0.2000, 35.1000),
    "Narok": (-1.0833, 35.8667),
    "Nyamira": (-0.5667, 34.9500),
    "Nyandarua": (-0.4179, 36.6674),
    "Nyeri": (-0.4201, 36.9476),
    "Samburu": (1.1000, 36.6667),
    "Siaya": (0.0600, 34.2900),
    "Taita-Taveta": (-3.3968, 38.3700),
    "Tana River": (-1.5069, 40.0089),
    "Tharaka-Nithi": (-0.2966, 37.8688),
    "Trans-Nzoia": (1.0567, 34.9500),
    "Turkana": (3.1167, 35.6000),
    "Uasin-Gishu": (0.5167, 35.2833),
    "Vihiga": (0.0761, 34.7198),
    "Wajir": (1.7488, 40.0582),
    "West Pokot": (1.9167, 35.1667)
}

@st.cache_data(ttl=3600, show_spinner="Fetching climate data from NASA...")
def fetch_nasa_data(lat: float, lon: float, county_name: str) -> pd.DataFrame:
    """
    Fetch climate data from NASA POWER API with robust error handling
    """
    try:
        # Construct API URL
        base_url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
        params = {
            "parameters": "T2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN",
            "community": "AG",
            "longitude": lon,
            "latitude": lat,
            "start": 2000,
            "end": 2024,
            "format": "JSON"
        }
        
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if "properties" not in data or "parameter" not in data["properties"]:
            st.error(f"No climate data available for {county_name}")
            return None
            
        parameters = data["properties"]["parameter"]
        
        # Create DataFrame
        dates = []
        temperatures = []
        rainfall = []
        solar_radiation = []
        
        for date_str, temp in parameters["T2M"].items():
            try:
                # Convert YYYYMM format to datetime
                year = int(date_str[:4])
                month = int(date_str[4:6])
                date_obj = datetime(year, month, 1)
                
                dates.append(date_obj)
                temperatures.append(temp)
                rainfall.append(parameters["PRECTOTCORR"][date_str])
                solar_radiation.append(parameters["ALLSKY_SFC_SW_DWN"][date_str])
                
            except (ValueError, KeyError):
                continue
        
        if not dates:
            st.warning(f"No valid data found for {county_name}")
            return None
            
        df = pd.DataFrame({
            "date": dates,
            "temperature": temperatures,
            "rainfall": rainfall,
            "solar_radiation": solar_radiation,
            "county": county_name
        })
        
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        
        return df
        
    except requests.exceptions.RequestException as e:
        st.error(f"Network error fetching data for {county_name}: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error for {county_name}: {str(e)}")
        return None

def analyze_climate_data(df: pd.DataFrame, county_name: str) -> dict:
    """
    Analyze climate data and generate insights
    """
    if df is None or df.empty:
        return None
        
    # Basic statistics
    avg_temp = df["temperature"].mean()
    avg_rainfall = df["rainfall"].mean()
    avg_solar = df["solar_radiation"].mean()
    
    # Calculate trends
    yearly_data = df.groupby("year").agg({
        "temperature": "mean",
        "rainfall": "mean"
    }).reset_index()
    
    if len(yearly_data) > 1:
        temp_trend = np.polyfit(yearly_data["year"], yearly_data["temperature"], 1)[0]
        rain_trend = np.polyfit(yearly_data["year"], yearly_data["rainfall"], 1)[0]
    else:
        temp_trend = 0
        rain_trend = 0
    
    # Risk assessment
    if temp_trend > 0.02:  # > 0.2°C per decade
        risk_level = "HIGH"
        risk_color = "🔴"
        recommendation = "Urgent need for heat adaptation measures"
    elif temp_trend > 0.01:  # > 0.1°C per decade
        risk_level = "MEDIUM"
        risk_color = "🟡" 
        recommendation = "Monitor trends and plan adaptation"
    else:
        risk_level = "LOW"
        risk_color = "🟢"
        recommendation = "Maintain current climate resilience efforts"
    
    return {
        "county": county_name,
        "avg_temperature": avg_temp,
        "avg_rainfall": avg_rainfall,
        "avg_solar_radiation": avg_solar,
        "temperature_trend_per_decade": temp_trend * 10,
        "rainfall_trend_per_decade": rain_trend * 10,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "recommendation": recommendation,
        "hottest_year": yearly_data.loc[yearly_data["temperature"].idxmax(), "year"],
        "wettest_year": yearly_data.loc[yearly_data["rainfall"].idxmax(), "year"]
    }

def create_comparison_plot(df1: pd.DataFrame, df2: pd.DataFrame, county1: str, county2: str):
    """
    Create comparison plots for two counties
    """
    fig = go.Figure()
    
    # Temperature comparison
    fig.add_trace(go.Scatter(
        x=df1["date"], y=df1["temperature"],
        name=f"{county1} Temperature",
        line=dict(color="#FF6B6B", width=2),
        yaxis="y1"
    ))
    
    fig.add_trace(go.Scatter(
        x=df2["date"], y=df2["temperature"], 
        name=f"{county2} Temperature",
        line=dict(color="#4ECDC4", width=2),
        yaxis="y1"
    ))
    
    fig.update_layout(
        title=f"Temperature Comparison: {county1} vs {county2}",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        hovermode="x unified",
        height=400
    )
    
    return fig

# Sidebar configuration
st.sidebar.title("🌦️ Configuration")

# County selection
county1 = st.sidebar.selectbox(
    "Select First County",
    options=list(COUNTIES.keys()),
    index=29  # Default to Nairobi
)

county2 = st.sidebar.selectbox(
    "Select Second County", 
    options=list(COUNTIES.keys()),
    index=0   # Default to Baringo
)

# Analysis type
analysis_type = st.sidebar.radio(
    "Analysis Type",
    ["Climate Comparison", "Single County Analysis", "Trend Analysis"]
)

# Main app logic
if st.sidebar.button("Analyze Climate Data", type="primary"):
    
    with st.spinner("Fetching and analyzing climate data..."):
        # Fetch data for both counties
        data1 = fetch_nasa_data(COUNTIES[county1][0], COUNTIES[county1][1], county1)
        data2 = fetch_nasa_data(COUNTIES[county2][0], COUNTIES[county2][1], county2)
    
    if data1 is not None and data2 is not None:
        
        # Analyze data
        analysis1 = analyze_climate_data(data1, county1)
        analysis2 = analyze_climate_data(data2, county2)
        
        if analysis1 and analysis2:
            # Display key metrics
            st.subheader("📊 Climate Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    f"{county1} Avg Temperature",
                    f"{analysis1['avg_temperature']:.1f}°C",
                    f"{analysis1['temperature_trend_per_decade']:+.2f}°C/decade"
                )
            
            with col2:
                st.metric(
                    f"{county2} Avg Temperature", 
                    f"{analysis2['avg_temperature']:.1f}°C",
                    f"{analysis2['temperature_trend_per_decade']:+.2f}°C/decade"
                )
            
            with col3:
                st.metric(
                    f"{county1} Avg Rainfall",
                    f"{analysis1['avg_rainfall']:.1f} mm/month"
                )
            
            with col4:
                st.metric(
                    f"{county2} Avg Rainfall",
                    f"{analysis2['avg_rainfall']:.1f} mm/month"
                )
            
            # Risk assessment
            st.subheader("🚨 Climate Risk Assessment")
            
            risk_col1, risk_col2 = st.columns(2)
            
            with risk_col1:
                st.info(f"""
                **{county1}**
                - Risk Level: {analysis1['risk_color']} {analysis1['risk_level']}
                - Trend: {analysis1['temperature_trend_per_decade']:+.2f}°C per decade
                - Recommendation: {analysis1['recommendation']}
                """)
            
            with risk_col2:
                st.info(f"""
                **{county2}**
                - Risk Level: {analysis2['risk_color']} {analysis2['risk_level']}
                - Trend: {analysis2['temperature_trend_per_decade']:+.2f}°C per decade  
                - Recommendation: {analysis2['recommendation']}
                """)
            
            # Visualizations
            st.subheader("📈 Climate Trends")
            
            tab1, tab2, tab3 = st.tabs(["Temperature", "Rainfall", "Solar Radiation"])
            
            with tab1:
                fig_temp = create_comparison_plot(data1, data2, county1, county2)
                st.plotly_chart(fig_temp, use_container_width=True)
            
            with tab2:
                fig_rain = go.Figure()
                fig_rain.add_trace(go.Scatter(
                    x=data1["date"], y=data1["rainfall"],
                    name=f"{county1} Rainfall",
                    line=dict(color="#45B7D1", width=2)
                ))
                fig_rain.add_trace(go.Scatter(
                    x=data2["date"], y=data2["rainfall"],
                    name=f"{county2} Rainfall", 
                    line=dict(color="#96CEB4", width=2)
                ))
                fig_rain.update_layout(
                    title=f"Rainfall Comparison: {county1} vs {county2}",
                    height=400
                )
                st.plotly_chart(fig_rain, use_container_width=True)
            
            with tab3:
                fig_solar = go.Figure()
                fig_solar.add_trace(go.Scatter(
                    x=data1["date"], y=data1["solar_radiation"],
                    name=f"{county1} Solar Radiation",
                    line=dict(color="#FFEAA7", width=2)
                ))
                fig_solar.add_trace(go.Scatter(
                    x=data2["date"], y=data2["solar_radiation"],
                    name=f"{county2} Solar Radiation",
                    line=dict(color="#FDCB6E", width=2)
                ))
                fig_solar.update_layout(
                    title=f"Solar Radiation Comparison: {county1} vs {county2}",
                    height=400
                )
                st.plotly_chart(fig_solar, use_container_width=True)
            
            # Detailed analysis
            st.subheader("🔍 Detailed Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**{county1} Climate Profile**")
                st.write(f"- Average Temperature: {analysis1['avg_temperature']:.1f}°C")
                st.write(f"- Average Rainfall: {analysis1['avg_rainfall']:.1f} mm/month")
                st.write(f"- Average Solar Radiation: {analysis1['avg_solar_radiation']:.1f} MJ/m²")
                st.write(f"- Hottest Year: {analysis1['hottest_year']}")
                st.write(f"- Wettest Year: {analysis1['wettest_year']}")
            
            with col2:
                st.write(f"**{county2} Climate Profile**")
                st.write(f"- Average Temperature: {analysis2['avg_temperature']:.1f}°C")
                st.write(f"- Average Rainfall: {analysis2['avg_rainfall']:.1f} mm/month")
                st.write(f"- Average Solar Radiation: {analysis2['avg_solar_radiation']:.1f} MJ/m²")
                st.write(f"- Hottest Year: {analysis2['hottest_year']}")
                st.write(f"- Wettest Year: {analysis2['wettest_year']}")
        
        else:
            st.error("Could not analyze climate data. Please try again.")
    
    else:
        st.error("Failed to fetch climate data. Please check your internet connection.")

else:
    # Welcome message
    st.info("""
    👆 **Get started by selecting two counties and clicking 'Analyze Climate Data'**
    
    This app provides:
    - 📊 Climate comparison between any two Kenyan counties
    - 🚨 AI-powered risk assessment and recommendations  
    - 📈 Interactive trend visualizations
    - 🌡️ Temperature, rainfall, and solar radiation analysis
    - 📅 Data from 2000 to 2024 from NASA POWER
    """)
    
    # Quick stats
    st.subheader("🇰🇪 About Kenyan Climate")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Counties Covered", "47", "All counties")
    
    with col2:
        st.metric("Data Period", "24 years", "2000-2024")
    
    with col3:
        st.metric("Climate Parameters", "3", "Temp, Rain, Solar")

# Footer
st.markdown("---")
st.markdown(
    "**Climate Mirror Kenya** | Built with Streamlit & NASA POWER Data | "
    "Data updated regularly"
)