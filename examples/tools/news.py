"""Google News scraping tools for fetching latest news headlines."""

import asyncio
import httpx
import re
from typing import List, Dict, Any, Optional
from xml.etree import ElementTree as ET
from miiflow_agent.core.tools import tool
from miiflow_agent.core.tools.http.proxy_utils import get_proxy_config, should_use_proxy
import urllib.parse

@tool("get_top_news", "Fetch top N news headlines from Google News RSS feed")
async def get_top_news(n: int = 5) -> str:
    """Fetch the latest news headlines from Google News."""
    if n < 1 or n > 20:
        return f"Invalid count: {n}. Valid range is 1-20."
    
    url = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"
    
    try:
        proxy_config = get_proxy_config()
        client_kwargs = {"timeout": 15.0}
        
        if proxy_config and should_use_proxy(url):
            client_kwargs["proxies"] = proxy_config
        
        async with httpx.AsyncClient(**client_kwargs) as client:
            response = await client.get(url)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            items = root.findall(".//item")
            
            if not items:
                return "No news articles found"
            
            news_items = []
            for item in items[:n]:
                title = item.find("title")
                title_text = title.text if title is not None else "No title"
                
                pub_date = item.find("pubDate")
                date_text = pub_date.text if pub_date is not None else "No date"
                
                source = item.find("source")
                source_text = source.text if source is not None else "Unknown source"
                
                # Clean up title (remove HTML entities)
                title_text = re.sub(r'&[a-zA-Z0-9]+;', '', title_text)
                
                news_items.append({
                    "title": title_text,
                    "source": source_text,
                    "date": date_text
                })
            
            # Format output
            result = f"📰 Top {len(news_items)} News Headlines:\n\n"
            for i, item in enumerate(news_items, 1):
                result += f"{i}. **{item['title']}**\n"
                result += f"   📍 Source: {item['source']}\n"
                result += f"    {item['date']}\n\n"
            
            return result.strip()
            
    except httpx.TimeoutException:
        return "Timeout while fetching news from Google News"
    except httpx.HTTPStatusError as e:
        return f"HTTP error {e.response.status_code} while fetching news"
    except ET.ParseError:
        return "Error parsing RSS feed from Google News"
    except Exception as e:
        return f"Error fetching news: {str(e)}"


@tool("search_news_by_topic", "Search Google News for specific topic or keyword")
async def search_news_by_topic(topic: str, n: int = 3) -> str:
    """Search Google News for articles related to a specific topic."""
    if not topic or len(topic.strip()) < 2:
        return "Topic must be at least 2 characters long"
    
    if n < 1 or n > 10:
        return f"Invalid count: {n}. Valid range is 1-10."
    
    encoded_topic = urllib.parse.quote_plus(topic.strip())
    url = f"https://news.google.com/rss/search?q={encoded_topic}&hl=en-US&gl=US&ceid=US:en"
    
    try:
        proxy_config = get_proxy_config()
        client_kwargs = {"timeout": 15.0}
        
        if proxy_config and should_use_proxy(url):
            client_kwargs["proxies"] = proxy_config
        
        async with httpx.AsyncClient(**client_kwargs) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            items = root.findall(".//item")
            
            if not items:
                return f"No news articles found for topic: {topic}"
            
            news_items = []
            for item in items[:n]:
                title = item.find("title")
                title_text = title.text if title is not None else "No title"
                
                description = item.find("description") 
                desc_text = description.text if description is not None else ""
                
                pub_date = item.find("pubDate")
                date_text = pub_date.text if pub_date is not None else "No date"
                
                source = item.find("source")
                source_text = source.text if source is not None else "Unknown source"
                
                title_text = re.sub(r'&[a-zA-Z0-9]+;', '', title_text)
                desc_text = re.sub(r'<[^>]+>', '', desc_text)
                desc_text = re.sub(r'&[a-zA-Z0-9]+;', '', desc_text)
                
                if len(desc_text) > 150:
                    desc_text = desc_text[:150] + "..."
                
                news_items.append({
                    "title": title_text,
                    "description": desc_text,
                    "source": source_text,
                    "date": date_text
                })
            
            result = f"🔍 News search results for '{topic}' ({len(news_items)} articles):\n\n"
            for i, item in enumerate(news_items, 1):
                result += f"{i}. **{item['title']}**\n"
                if item['description']:
                    result += f"   📝 {item['description']}\n"
                result += f"   📍 Source: {item['source']}\n"
                result += f"   📅 {item['date']}\n\n"
            
            return result.strip()
            
    except httpx.TimeoutException:
        return f"Timeout while searching news for topic: {topic}"
    except httpx.HTTPStatusError as e:
        return f"HTTP error {e.response.status_code} while searching news for topic: {topic}"
    except ET.ParseError:
        return f"Error parsing RSS feed for topic: {topic}"
    except Exception as e:
        return f"Error searching news for topic '{topic}': {str(e)}"


__all__ = ['get_top_news', 'search_news_by_topic']
