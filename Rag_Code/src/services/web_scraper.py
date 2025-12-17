"""Web Scraping Service for fallback data retrieval"""
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
from src.config import Config
import logging
import time

logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def search_and_scrape(self, query: str, max_results: int = Config.MAX_WEB_RESULTS) -> List[Dict]:
        """Search web and scrape relevant content"""
        results = []
        
        # Use DuckDuckGo or Google search (simplified - in production use proper search API)
        search_urls = self._get_search_urls(query)
        
        for url in search_urls[:max_results]:
            try:
                content = self._scrape_url(url)
                if content:
                    results.append({
                        'content': content,
                        'source': url,
                        'title': self._extract_title(url)
                    })
                time.sleep(1)  # Be respectful
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                continue
        
        return results
    
    def _get_search_urls(self, query: str) -> List[str]:
        """Get search result URLs (simplified - use proper search API in production)"""
        # For demo purposes, return some relevant URLs
        # In production, integrate with Google Custom Search API or similar
        query_lower = query.lower()
        
        # Map common topics to relevant URLs
        url_map = {
            'data structure': [
                'https://en.wikipedia.org/wiki/Data_structure',
                'https://www.geeksforgeeks.org/data-structures/',
            ],
            'artificial intelligence': [
                'https://en.wikipedia.org/wiki/Artificial_intelligence',
                'https://www.ibm.com/topics/artificial-intelligence',
            ],
            'machine learning': [
                'https://en.wikipedia.org/wiki/Machine_learning',
                'https://www.ibm.com/topics/machine-learning',
            ],
            'rag': [
                'https://en.wikipedia.org/wiki/Retrieval-augmented_generation',
                'https://www.pinecone.io/learn/retrieval-augmented-generation/',
            ],
            'langchain': [
                'https://www.langchain.com/',
                'https://python.langchain.com/docs/get_started/introduction',
            ]
        }
        
        urls = []
        for key, url_list in url_map.items():
            if key in query_lower:
                urls.extend(url_list)
        
        # If no match, return some default URLs
        if not urls:
            urls = [
                'https://en.wikipedia.org/wiki/Artificial_intelligence',
                'https://en.wikipedia.org/wiki/Machine_learning'
            ]
        
        return urls[:Config.MAX_WEB_RESULTS]
    
    def _scrape_url(self, url: str) -> str:
        """Scrape content from URL"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text[:1000]  # Limit to 1000 chars
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {e}")
            return None
    
    def _extract_title(self, url: str) -> str:
        """Extract title from URL"""
        try:
            response = self.session.get(url, timeout=5)
            soup = BeautifulSoup(response.content, 'html.parser')
            title = soup.find('title')
            return title.text.strip() if title else url
        except:
            return url

