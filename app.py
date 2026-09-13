import json
import requests
import streamlit as st
from groq import Groq

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Run Weather AI Agent",
    page_icon="🏃",
    layout="centered"
)

st.title("🏃 Running Weather Safety Agent")
st.write("Get real-time weather safety evaluations and your recommended maximum running duration.")

# ---------------------------------------------------------------------------
# API Key Handling & Settings (Sidebar)
# ---------------------------------------------------------------------------
# 1. Try to load key from Streamlit Cloud Secrets (GROQ_API_KEY)
secret_key = st.secrets.get("GROQ_API_KEY", "")

with st.sidebar:
    st.header("Settings")
    
    # If key is not in Secrets, allow manual input
    if secret_key:
        st.success("API Key loaded from Secrets!")
        groq_key = secret_key
    else:
        groq_key = st.text_input("Groq API Key (Free)", type="password", help="Get a free key at console.groq.com")
    
    workout_type = st.selectbox(
        "Workout Type",
        ["Easy Run", "Tempo / Workout", "Long Run", "Interval Training"]
    )

# ---------------------------------------------------------------------------
# Tools: Geocoding & Open-Meteo Weather API
# ---------------------------------------------------------------------------
def geocode_city(city_name: str):
    """Converts a city name to latitude, longitude, and formatted name."""
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
    try:
        res = requests.get(geo_url, timeout=10).json()
        if not res.get("results"):
            return None
        loc = res["results"][0]
        return loc["latitude"], loc["longitude"], loc["name"], loc.get("country", "")
    except Exception:
        return None

def get_current_weather(latitude: float, longitude: float) -> str:
    """Fetches real-time weather metrics from Open-Meteo API."""
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}"
        f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,"
        f"precipitation,wind_speed_10m,wind_gusts_10m"
        f"&timezone=auto"
    )
    try:
        res = requests.get(weather_url, timeout=10).json()["current"]
        weather_summary = {
            "temperature_c": res["temperature_2m"],
            "feels_like_c": res["apparent_temperature"],
            "humidity_percent": res["relative_humidity_2m"],
            "precipitation_mm": res["precipitation"],
            "wind_speed_kmh": res["wind_speed_10m"],
            "wind_gusts_kmh": res["wind_gusts_10m"],
        }
        return json.dumps(weather_summary)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch weather: {str(e)}"})

# Function schema passed to the LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get current outdoor weather metrics (temperature, humidity, wind, precipitation) by latitude and longitude.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "Latitude coordinate"},
                    "longitude": {"type": "number", "description": "Longitude coordinate"},
                },
                "required": ["latitude", "longitude"],
            },
        },
    }
]

# System prompt defining running duration safety guidelines
SYSTEM_PROMPT = """You are an expert running coach and outdoor sports safety advisor.
When asked if it is safe to run:
1. Call the weather tool to get current live conditions.
2. Evaluate conditions against safety standards:
   - **Optimal (10°C–18°C / 50°F–65°F)**: Unrestricted / Full Long Run (2+ hrs).
   - **Mild Caution (18°C–25°C OR 0°C–10°C)**: Max 60–90 minutes.
   - **Moderate Caution (25°C–30°C OR -7°C–0°C)**: Max 30–45 minutes.
   - **High Risk (30°C–35°C OR -18°C–-7°C)**: Max 15–30 minutes (easy recovery only).
   - **Extreme Danger (>35°C OR < -18°C OR severe storm/lightening)**: 0 minutes (NO-GO).

3. Structure your final output clearly:
   - **Verdict**: [🟢 GO / 🟡 CAUTION / 🔴 NO-GO]
   - **Max Safe Duration**: [e.g., 45 minutes max / Unlimited]
   - **Current Conditions**: Temperature (Feels like), Humidity, Wind speed, Precipitation.
   - **Coach Advice & Gear**: Hydration, pacing changes, and what to wear.
"""

# ---------------------------------------------------------------------------
# Main App User Interface
# ---------------------------------------------------------------------------
location = st.text_input("Enter your location:", value="St. Charles")

if st.button("Check Running Conditions", type="primary"):
    if not groq_key:
        st.error("Missing Groq API Key! Please enter your key in the sidebar or add GROQ_API_KEY to Streamlit Secrets.")
    else:
        client = Groq(api_key=groq_key)
        
        with st.spinner("Finding location coordinates..."):
            coords = geocode_city(location)
            
        if not coords:
            st.error(f"Could not find coordinates for location '{location}'. Please check the spelling.")
        else:
            lat, lon, city_name, country = coords
            st.caption(f"📍 Checking weather for **{city_name}, {country}** ({lat:.2f}, {lon:.2f})")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Can I go for a {workout_type} in {city_name}, {country} right now? What is my max safe run time?"}
            ]

            with st.spinner("Evaluating weather safety with AI..."):
                try:
                    # Initial completion call
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=messages,
                        tools=tools,
                        tool_choice="auto"
                    )
                    
                    response_msg = response.choices[0].message

                    # Handle Tool Call Execution
                    if response_msg.tool_calls:
                        messages.append(response_msg)
                        
                        for tool_call in response_msg.tool_calls:
                            if tool_call.function.name == "get_current_weather":
                                args = json.loads(tool_call.function.arguments)
                                weather_data = get_current_weather(args["latitude"], args["longitude"])
                                
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": weather_data
                                })

                        # Final completion call with tool data
                        final_response = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=messages
                        )
                        st.markdown(final_response.choices[0].message.content)
                    else:
                        st.markdown(response_msg.content)

                except Exception as e:
                    st.error(f"An error occurred while connecting to Groq: {str(e)}")
