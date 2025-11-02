import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# -----------------------------------------------
# APP TITLE AND INTRODUCTION
# -----------------------------------------------
st.set_page_config(page_title="Climate Mirror Kenya", layout="wide")

st.title("🌍 Clime360 - Climate Mirror Kenya")
st.markdown("""
**Explore NASA POWER Climate Data (2000–2024)**  
Compare temperature, rainfall, and solar radiation for any two Kenyan counties.  
Data source: [NASA POWER API](https://power.larc.nasa.gov/).
""")

# -----------------------------------------------
# COUNTY DATA (Approximate Coordinates)
# -----------------------------------------------
counties = {
    "West Pokot": (-1.296, 35.12),
    "Nairobi": (-1.29, 36.82),
    "Mombasa": (-4.04, 39.67),
    "Kisumu": (-0.09, 34.75),
    "Garissa": (-0.45, 39.64),
    "Turkana": (3.12, 35.6),
    "Narok": (-1.08, 35.87),
    "Machakos": (-1.51, 37.26),
    "Meru": (0.05, 37.65),
    "Eldoret": (0.52, 35.27),
}

# -----------------------------------------------
# NASA DATA FETCH FUNCTION
# -----------------------------------------------
@st.cache_data(ttl=3600)
def get_nasa_data(lat, lon):
    """Fetch NASA POWER climate data with caching"""
    url = f"https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=T2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN&community=AG&longitude={lon}&latitude={lat}&start=2000&end=2024&format=JSON"
    try:
        res = requests.get(url, timeout=30)
        if res.status_code != 200:
            return None

        data = res.json()["properties"]["parameter"]
        df = pd.DataFrame({
            "Date": list(data["T2M"].keys()),
            "Temperature_C": list(data["T2M"].values()),
            "Rainfall_mm": list(data["PRECTOTCORR"].values()),
            "SolarRadiation_MJ": list(data["ALLSKY_SFC_SW_DWN"].values())
        })

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", format="mixed")
        df = df.dropna(subset=["Date"])
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# -----------------------------------------------
# CLIMATE ANALYSIS FUNCTIONS
# -----------------------------------------------
def analyze_climate_trends(df, county_name):
    """Generate climate insights from the data"""
    if df is None or len(df) == 0:
        return {}
    
    # Split into periods for comparison
    early_period = df[df["Year"] <= 2010]
    recent_period = df[df["Year"] >= 2015]
    
    analysis = {
        "county": county_name,
        "avg_temp_early": early_period["Temperature_C"].mean(),
        "avg_temp_recent": recent_period["Temperature_C"].mean(),
        "avg_rainfall_early": early_period["Rainfall_mm"].mean(),
        "avg_rainfall_recent": recent_period["Rainfall_mm"].mean(),
        "temp_change": recent_period["Temperature_C"].mean() - early_period["Temperature_C"].mean(),
        "rainfall_change": recent_period["Rainfall_mm"].mean() - early_period["Rainfall_mm"].mean(),
        "hottest_year": df.groupby("Year")["Temperature_C"].mean().idxmax(),
        "wettest_year": df.groupby("Year")["Rainfall_mm"].sum().idxmax(),
        "driest_year": df.groupby("Year")["Rainfall_mm"].sum().idxmin(),
    }
    
    return analysis

def generate_awareness_message(analysis1, analysis2):
    """Generate climate awareness message based on data"""
    if not analysis1 or not analysis2:
        return ""
    
    message = f"""
### 🌡️ Climate Change Insights ({analysis1['county']} vs {analysis2['county']})

**Temperature Trends:**
- {analysis1['county']}: {analysis1['temp_change']:+.2f}°C change (2000-2010 vs 2015-2024)
- {analysis2['county']}: {analysis2['temp_change']:+.2f}°C change (2000-2010 vs 2015-2024)

**Rainfall Patterns:**
- {analysis1['county']}: {analysis1['rainfall_change']:+.1f}mm monthly change
- {analysis2['county']}: {analysis2['rainfall_change']:+.1f}mm monthly change

**Key Observations:**
- Hottest years: {analysis1['county']} ({analysis1['hottest_year']}), {analysis2['county']} ({analysis2['hottest_year']})
- Climate variability is increasing across Kenya
- Extreme weather events are becoming more frequent

### 📢 Share This Message:
```
Climate data shows Kenya is warming! 
{analysis1['county']} has seen {abs(analysis1['temp_change']):.2f}°C temperature change since 2000.
We must act now for climate resilience. #ClimateActionKE #Clime360
Source: NASA POWER Data
```
"""
    return message

# -----------------------------------------------
# USER INPUTS
# -----------------------------------------------
col1, col2 = st.columns(2)
with col1:
    county1 = st.selectbox("Select County 1", list(counties.keys()), index=0)
with col2:
    county2 = st.selectbox("Select County 2", list(counties.keys()), index=1)

# -----------------------------------------------
# FETCH AND COMPARE DATA
# -----------------------------------------------
with st.spinner("Fetching NASA climate data..."):
    data1 = get_nasa_data(*counties[county1])
    data2 = get_nasa_data(*counties[county2])

if data1 is not None and data2 is not None:
    
    # Generate analysis
    analysis1 = analyze_climate_trends(data1, county1)
    analysis2 = analyze_climate_trends(data2, county2)
    
    # Display awareness message
    st.markdown(generate_awareness_message(analysis1, analysis2))
    
    # Tabs for detailed views
    tab1, tab2, tab3 = st.tabs(["📊 Climate Comparison", "📈 Long-Term Trends", "📋 Data Summary"])

    # Tab 1: Climate Comparison
    with tab1:
        st.subheader(f"Comparing {county1} vs {county2} (2000–2024)")

        for metric, label, color1, color2 in [
            ("Temperature_C", "Temperature (°C)", "#FF6B6B", "#4ECDC4"),
            ("Rainfall_mm", "Rainfall (mm)", "#45B7D1", "#96CEB4"),
            ("SolarRadiation_MJ", "Solar Radiation (MJ/m²)", "#FFEAA7", "#DFE6E9")
        ]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data1["Date"], y=data1[metric],
                mode="lines", name=county1,
                line=dict(color=color1, width=2)
            ))
            fig.add_trace(go.Scatter(
                x=data2["Date"], y=data2[metric],
                mode="lines", name=county2,
                line=dict(color=color2, width=2)
            ))
            fig.update_layout(
                title=f"{label}: {county1} vs {county2}",
                xaxis_title="Date",
                yaxis_title=label,
                hovermode="x unified",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Long-Term Trends
    with tab2:
        st.subheader("📈 Long-Term Climate Trends (2000–2024)")
        
        # Temperature trends
        trend1 = data1.groupby("Year")[["Temperature_C", "Rainfall_mm"]].mean().reset_index()
        trend2 = data2.groupby("Year")[["Temperature_C", "Rainfall_mm"]].mean().reset_index()
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            fig_temp = go.Figure()
            fig_temp.add_trace(go.Scatter(
                x=trend1["Year"], y=trend1["Temperature_C"],
                mode="lines+markers", name=county1,
                line=dict(color="#FF6B6B", width=3)
            ))
            fig_temp.add_trace(go.Scatter(
                x=trend2["Year"], y=trend2["Temperature_C"],
                mode="lines+markers", name=county2,
                line=dict(color="#4ECDC4", width=3)
            ))
            fig_temp.update_layout(
                title="Annual Average Temperature",
                xaxis_title="Year",
                yaxis_title="Temperature (°C)",
                height=400
            )
            st.plotly_chart(fig_temp, use_container_width=True)
        
        with col_b:
            fig_rain = go.Figure()
            fig_rain.add_trace(go.Scatter(
                x=trend1["Year"], y=trend1["Rainfall_mm"],
                mode="lines+markers", name=county1,
                line=dict(color="#45B7D1", width=3)
            ))
            fig_rain.add_trace(go.Scatter(
                x=trend2["Year"], y=trend2["Rainfall_mm"],
                mode="lines+markers", name=county2,
                line=dict(color="#96CEB4", width=3)
            ))
            fig_rain.update_layout(
                title="Annual Average Rainfall",
                xaxis_title="Year",
                yaxis_title="Rainfall (mm)",
                height=400
            )
            st.plotly_chart(fig_rain, use_container_width=True)

    # Tab 3: Data Summary
    with tab3:
        st.subheader("📊 Climate Statistics Summary")
        
        col_x, col_y = st.columns(2)
        
        with col_x:
            st.markdown(f"#### {county1}")
            st.metric("Avg Temperature (2000-2024)", f"{data1['Temperature_C'].mean():.1f}°C")
            st.metric("Total Annual Rainfall", f"{data1.groupby('Year')['Rainfall_mm'].sum().mean():.0f} mm")
            st.metric("Avg Solar Radiation", f"{data1['SolarRadiation_MJ'].mean():.1f} MJ/m²")
        
        with col_y:
            st.markdown(f"#### {county2}")
            st.metric("Avg Temperature (2000-2024)", f"{data2['Temperature_C'].mean():.1f}°C")
            st.metric("Total Annual Rainfall", f"{data2.groupby('Year')['Rainfall_mm'].sum().mean():.0f} mm")
            st.metric("Avg Solar Radiation", f"{data2['SolarRadiation_MJ'].mean():.1f} MJ/m²")

else:
    st.error("⚠️ Could not fetch data from NASA. Please check your connection or try again later.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and NASA POWER data | Climate Mirror Kenya")