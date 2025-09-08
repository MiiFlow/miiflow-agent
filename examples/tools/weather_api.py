"""Weather API tools using OpenWeatherMap."""

import os
from typing import Optional, Dict, Any
import httpx
from miiflow_llm.core.tools import tool

API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")


@tool("get_current_weather", "Get current weather for a city")
async def get_current_weather(
    city: str, 
    country_code: Optional[str] = None,
    units: str = "metric"
) -> Dict[str, Any]:
    
    if not API_KEY:
        raise ValueError("OPENWEATHERMAP_API_KEY environment variable required")
    
    q = f"{city},{country_code}" if country_code else city
    params = {"q": q, "appid": API_KEY, "units": units}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params=params,
            timeout=10.0
        )
        response.raise_for_status()
        return response.json()


@tool("get_weather_forecast", "Get 5-day weather forecast")
async def get_weather_forecast(
    city: str,
    country_code: Optional[str] = None,
    units: str = "metric"
) -> Dict[str, Any]:
    
    if not API_KEY:
        raise ValueError("OPENWEATHERMAP_API_KEY environment variable required")
    
    q = f"{city},{country_code}" if country_code else city
    params = {"q": q, "appid": API_KEY, "units": units}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.openweathermap.org/data/2.5/forecast",
            params=params,
            timeout=10.0
        )
        response.raise_for_status()
        return response.json()


@tool("get_weather_by_coords", "Get weather by coordinates")
async def get_weather_by_coords(
    lat: float,
    lon: float,
    units: str = "metric"
) -> Dict[str, Any]:
    
    if not API_KEY:
        raise ValueError("OPENWEATHERMAP_API_KEY environment variable required")
    
    params = {"lat": lat, "lon": lon, "appid": API_KEY, "units": units}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params=params,
            timeout=10.0
        )
        response.raise_for_status()
        return response.json()
