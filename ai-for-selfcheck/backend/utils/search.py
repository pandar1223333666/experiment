from typing import List, Optional, Dict, Any
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import tool
from config import settings
from schemas import SearchResult, SearchResponse

class SearchManager:
    def __init__(self):
        self.settings = settings
        self._search_wrapper = None
        self._initialize_search()
    
    def _initialize_search(self):
        search_params = {
            "engine": self.settings.search_engine,
            "gl": self.settings.search_country,
            "hl": self.settings.search_language,
            "num": self.settings.max_search_results
        }
        
        self._search_wrapper = SerpAPIWrapper(
            serpapi_api_key=self.settings.serpapi_api_key,
            params=search_params
        )            
    
    def is_available(self) -> bool:
        return self._search_wrapper is not None
    
    def search(self, query: str, max_results: Optional[int] = None) -> SearchResponse:
        raw_results = self._search_wrapper.run(query)
        
        search_results = self._parse_search_results(raw_results, max_results or self.settings.max_search_results)
        
        response = SearchResponse(
            query=query,
            results=search_results,
            total_results=len(search_results)
        )
        
        return response
            
    def _parse_search_results(self, raw_results: str, max_results: int) -> List[SearchResult]:
        results = []
        
        # 如果是字符串结果，尝试分割和解析
        if isinstance(raw_results, str):
            lines = raw_results.split('\n')
            for i, line in enumerate(lines[:max_results]):
                if line.strip():
                    result = SearchResult(
                        title=f"Search Result {i+1}",
                        url="",
                        snippet=line.strip(),
                        rank=i+1
                    )
                    results.append(result)
        
        # 如果是结构化数据
        elif isinstance(raw_results, list):
            for i, item in enumerate(raw_results[:max_results]):
                if isinstance(item, dict):
                    result = SearchResult(
                        title=item.get('title', f'Result {i+1}'),
                        url=item.get('link', ''),
                        snippet=item.get('snippet', ''),
                        rank=i+1
                    )
                    results.append(result)
        
        return results

search_manager = SearchManager()

@tool
def web_search_tool(query: str) -> str:
      response = search_manager.search(query)
      
      if not response.results:
         return f"未找到关于 '{query}' 的相关信息。"
      
      formatted_results = []
      for result in response.results[:5]: 
         formatted_results.append(f"• {result.snippet}")
      
      return f"关于 '{query}' 的搜索结果：\n" + "\n".join(formatted_results)

def get_search_tools() -> list:
   return [web_search_tool]


