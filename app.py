import json
import requests
import streamlit as st
from groq import Groq

st.set_page_config(page_title="Run Weather AI Agent", page_icon="🏃")

st.title("🏃 Running Weather Safety Agent")
st.write("Check real-time weather safety and get your maximum run duration.")

# Sidebar for Groq API Key & Settings
with st.sidebar:
    api_key = st.text_input("Groq API Key (Free)", type="password")
    workout_type = st.selectbox(
        "Workout Type",
        ["Easy Run", "Tempo / Workout", "Long Run", "Interval Training"]
    )

# Weather API function
def get_current_weather(latitude: float, longitude: float) -> str:
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}"
        f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,"
        f"precipitation,wind_speed_10m,wind_gusts_10m"
        f"&timezone=auto"
    )
    res = requests.get(weather_url, timeout=10).json()["current"]
    return json.dumps({
        "temperature_c": res["temperature_2m"],
        "feels_like_c": res["apparent_temperature"],
        "humidity_percent": res["relative_humidity_2m"],
        "precipitation_mm": res["precipitation"],
        "wind_speed_kmh": res["wind_speed_10m"]
    })

def geocode_city(city_name: str):
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
    res = requests.get(geo_url, timeout=10).json()
    if not res.get("results"):
        return None
    loc = res["results"][0]
    return loc["latitude"], loc["longitude"], loc["name"], loc.get("country", "")

# User Location Input
location = st.text_input("Enter your city:", value="St. Charles")

if st.button("Check Conditions"):
    if not api_key:
        st.error("Please enter your free Groq API key in the sidebar.")
    else:
        client = Groq(api_key=api_key)
        coords = geocode_city(location)
        
        if not coords:
            st.error("Location not found.")
        else:
            lat, lon, city_name, country = coords
            
            tools = [{
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Fetch current weather metrics",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "latitude": {"type": "number"},
                            "longitude": {"type": "number"}
                        },
                        "required": ["latitude", "longitude"]
                    }
                }
            }]
            
            system_prompt = """You are an expert running coach.
            Evaluate weather metrics to determine:
            1. Verdict (🟢 GO / 🟡 CAUTION / 🔴 NO-GO)
            2. Recommended Max Duration (e.g. 30 min, 60 min, Unrestricted)
            3. Weather summary & Pacing / Gear advice."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Can I go for a {workout_type} in {city_name}, {country}? How long can I run?"}
            ]

            with st.spinner("Fetching weather and evaluating safety..."):
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    tools=tools
                )
                
                response_msg = response.choices[0].message
                if response_msg.tool_calls:
                    messages.append(response_msg)
                    for tool in response_msg.tool_calls:
                        weather_data = get_current_weather(lat, lon)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool.id,
                            "content": weather_data
                        })
                    
                    final = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=messages
                    )
                    st.markdown(final.choices[0].message.content)
