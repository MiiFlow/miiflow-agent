"""JSONPlaceholder API tools for fetching posts and user data."""

import asyncio
import httpx
from typing import List, Dict, Any, Optional
from miiflow_agent.core.tools import tool
from miiflow_agent.core.tools.http.proxy_utils import get_proxy_config, should_use_proxy


@tool("get_user_posts", "Fetch posts for a specific user from JSONPlaceholder API")
async def get_user_posts(userId: int) -> str:
    if userId < 1 or userId > 10:
        return f"Invalid user ID: {userId}. Valid range is 1-10."
    
    url = "https://jsonplaceholder.typicode.com/posts"
    
    try:
        # Configure proxy if available and needed
        proxy_config = get_proxy_config()
        client_kwargs = {"timeout": 10.0}
        
        if proxy_config and should_use_proxy(url):
            client_kwargs["proxies"] = proxy_config
        
        async with httpx.AsyncClient(**client_kwargs) as client:
            response = await client.get(url, params={"userId": userId})
            response.raise_for_status()
            
            posts = response.json()
            
            if not posts:
                return f"No posts found for user {userId}"
            
            formatted_posts = []
            for post in posts[:5]: #sample test only, taking only first 5
                title = post.get('title', 'No title')
                body = post.get('body', 'No content')
                if len(body) > 100:
                    body = body[:100] + "..."
                
                formatted_posts.append(f"**{title}**\n   {body}")
            
            result = f"Here are the posts for user {userId}:\n\n"
            for i, post in enumerate(formatted_posts, 1):
                result += f"{i}. {post}\n\n"
            
            if len(posts) > 5:
                result += f"... and {len(posts) - 5} more posts"
            
            return result.strip()
            
    except httpx.TimeoutException:
        return f"Timeout while fetching posts for user {userId}"
    except httpx.HTTPStatusError as e:
        return f"HTTP error {e.response.status_code} while fetching posts for user {userId}"
    except Exception as e:
        return f"Error fetching posts for user {userId}: {str(e)}"


@tool("get_all_users", "Fetch all users from JSONPlaceholder API")
async def get_all_users() -> str:
    
    url = "https://jsonplaceholder.typicode.com/users"
    
    try:
        # Configure proxy if available and needed
        proxy_config = get_proxy_config()
        client_kwargs = {"timeout": 10.0}
        
        if proxy_config and should_use_proxy(url):
            client_kwargs["proxies"] = proxy_config
        
        async with httpx.AsyncClient(**client_kwargs) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            users = response.json()
            
            if not users:
                return "No users found"
            
            result = "Available users:\n\n"
            for user in users:
                user_id = user.get('id', 'Unknown')
                name = user.get('name', 'Unknown')
                email = user.get('email', 'No email')
                company = user.get('company', {}).get('name', 'No company')
                
                result += f"{user_id}. **{name}** ({email})\n   Company: {company}\n\n"
            
            return result.strip()
            
    except httpx.TimeoutException:
        return "Timeout while fetching users"
    except httpx.HTTPStatusError as e:
        return f"HTTP error {e.response.status_code} while fetching users"
    except Exception as e:
        return f"Error fetching users: {str(e)}"


@tool("get_post_comments", "Fetch comments for a specific post")
async def get_post_comments(postId: int) -> str:
    if postId < 1 or postId > 100:
        return f"Invalid post ID: {postId}. Valid range is 1-100."
    
    url = f"https://jsonplaceholder.typicode.com/posts/{postId}/comments"
    
    try:
        # Configure proxy if available and needed
        proxy_config = get_proxy_config()
        client_kwargs = {"timeout": 10.0}
        
        if proxy_config and should_use_proxy(url):
            client_kwargs["proxies"] = proxy_config
        
        async with httpx.AsyncClient(**client_kwargs) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            comments = response.json()
            
            if not comments:
                return f"No comments found for post {postId}"
            
            result = f"Comments for post {postId}:\n\n"
            for i, comment in enumerate(comments[:3], 1):  # Limit to first 3 comments
                name = comment.get('name', 'Anonymous')
                email = comment.get('email', 'No email')
                body = comment.get('body', 'No content')
                if len(body) > 80:
                    body = body[:80] + "..."
                
                result += f"{i}. **{name}** ({email})\n   {body}\n\n"
            
            if len(comments) > 3:
                result += f"... and {len(comments) - 3} more comments"
            
            return result.strip()
            
    except httpx.TimeoutException:
        return f"Timeout while fetching comments for post {postId}"
    except httpx.HTTPStatusError as e:
        return f"HTTP error {e.response.status_code} while fetching comments for post {postId}"
    except Exception as e:
        return f"Error fetching comments for post {postId}: {str(e)}"

#backward compatibility, will be deprecated soon.
__all__ = ['get_user_posts', 'get_all_users', 'get_post_comments']
