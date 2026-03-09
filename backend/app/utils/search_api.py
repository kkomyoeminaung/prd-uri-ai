"""
Google Search API wrapper
"""

import aiohttp
import json
import os
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class GoogleSearchAPI:
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
        self.cx = os.getenv('GOOGLE_SEARCH_CX')
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    async def search(self, query: str, num_results: int = 10) -> List[Dict]:
        if not self.api_key or not self.cx:
            logger.warning("Google Search API keys not configured, using mock results")
            return self._mock_search(query, num_results)
        
        params = {
            'key': self.api_key,
            'cx': self.cx,
            'q': query,
            'num': min(num_results, 10)
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = []
                        for item in data.get('items', []):
                            results.append({
                                'title': item.get('title'),
                                'url': item.get('link'),
                                'snippet': item.get('snippet'),
                                'source': 'google'
                            })
                        return results
                    else:
                        logger.error(f"Search API error: {response.status}")
                        return self._mock_search(query, num_results)
        except Exception as e:
            logger.error(f"Search exception: {e}")
            return self._mock_search(query, num_results)
    
    def _mock_search(self, query: str, num_results: int) -> List[Dict]:
        return [
            {
                'title': f"Wikipedia: {query}",
                'url': f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                'snippet': f"Comprehensive information about {query}",
                'source': 'mock'
            },
            {
                'title': f"Britannica: {query}",
                'url': f"https://www.britannica.com/topic/{query.replace(' ', '-')}",
                'snippet': f"Encyclopedia article about {query}",
                'source': 'mock'
            },
            {
                'title': f"Latest research on {query}",
                'url': f"https://arxiv.org/search/?query={query.replace(' ', '+')}",
                'snippet': f"Recent papers and preprints about {query}",
                'source': 'mock'
            }
        ][:num_results]
