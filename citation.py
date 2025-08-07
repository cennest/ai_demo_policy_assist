from typing  import List
from pydantic import BaseModel


class Citation(BaseModel):
    """Citation model for tracking document references"""
    score: float
    start_index: int
    end_index: int
    text: str 
    segments: List['Segments']    
   

class Segments(BaseModel):
    title: str
    chunk_index: int
    link: str
    original_link:str | None = None 
    placeholder: str | None = None 

    def get_formated_link(self) -> str:
        """Format citation as markdown link"""
        return f"[[{self.chunk_index+1}]({self.original_link})]"  #if self.chunk_index >= 0 else ""    

    def get_placeholder(self) -> str:
        """Format citation as markdown link"""
        return f" {self.placeholder}" 

