import asyncio
from collections import defaultdict
import json
import re
import textwrap
import hashlib
import urllib.parse
from typing import List, Optional, Any, Dict, Tuple, Union
from pathlib import Path
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig
from google.genai.types import Tool, GoogleSearch, UrlContext, Part, File
import httpx


class GoogleGenAI:
    """
    Handy wrapper for Google Gemini API operations including:
    - Content generation (with optional file support)
    - Citation extraction and formatting
    - Structured output with Pydantic models
    """
    
    def __init__(self, api_key: str, default_model_id: str = "gemini-2.5-flash"):
        """
        Initialize Gemini wrapper
        
        Args:
            api_key: Google API key (if not provided, will prompt or use env var)
            default_model_id: Default Gemini model to use (can be overridden per call)
        """
        self.default_model_id = default_model_id
        self.client = genai.Client(api_key=api_key)
        self.cache_dir = Path("__cache__")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Clean up old cache files on initialization
        self.cleanup_cache()
    
    async def _find_existing_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """Check if a file with the same name and size already exists on Google AI"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return None
        
        try:
            # Get local file stats
            local_size = file_path.stat().st_size
            local_name = file_path.name
            
            # List files from Google AI
            files_response = await self.client.aio.files.list()
            
            # Check for existing file with same name and size
            for file_info in files_response.files:
                if (file_info.display_name == local_name and 
                    hasattr(file_info, 'size_bytes') and 
                    file_info.size_bytes == local_size):
                    return file_info.uri
                    
        except Exception as e:
            # If listing fails, proceed with upload
            print(f"Warning: Could not check existing files: {e}")
            
        return None
    
    def _find_existing_file_sync(self, file_path: Union[str, Path]) -> Optional[str]:
        """Check if a file with the same name and size already exists on Google AI (sync)"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return None
        
        try:
            # Get local file stats
            local_size = file_path.stat().st_size
            local_name = file_path.name
            
            # List files from Google AI
            files_response = self.client.files.list()
            
            # Check for existing file with same name and size
            for file_info in files_response.files:
                if (file_info.display_name == local_name and 
                    hasattr(file_info, 'size_bytes') and 
                    file_info.size_bytes == local_size):
                    return file_info.uri
                    
        except Exception as e:
            # If listing fails, proceed with upload
            print(f"Warning: Could not check existing files: {e}")
            
        return None
    
    def _get_file_extension(self, url: str, content_type: Optional[str] = None) -> str:
        """Determine file extension from URL or content type"""
        # Try to get extension from URL first
        parsed_url = urllib.parse.urlparse(url)
        path = parsed_url.path
        if '.' in path:
            ext = Path(path).suffix
            if ext:
                return ext
        
        # Fallback to content type
        if content_type:
            content_type_map = {
                'text/html': '.html',
                'text/plain': '.txt',
                'text/markdown': '.md',
                'text/csv': '.csv',
                'application/json': '.json',
                'application/pdf': '.pdf',
                'application/xml': '.xml',
                'text/xml': '.xml',
                'image/png': '.png',
                'image/jpeg': '.jpg',
                'image/gif': '.gif',
                'video/mp4': '.mp4',
                'audio/mp3': '.mp3',
                'audio/wav': '.wav'
            }
            return content_type_map.get(content_type, '.txt')
        
        return '.txt'  # Default extension
    
    def _create_cache_filename(self, url: str, content: bytes, extension: str) -> str:
        """Create a unique cache filename based on URL and content hash"""
        # Create hash from URL and content
        content_hash = hashlib.md5(url.encode() + content).hexdigest()[:12]
        # Clean URL for filename
        clean_url = re.sub(r'[^\w\-.]', '_', urllib.parse.urlparse(url).netloc)
        return f"{clean_url}_{content_hash}{extension}"
    
    def cleanup_cache(self, max_age_hours: int = 24) -> None:
        """Clean up old cache files older than max_age_hours"""
        import time
        
        if not self.cache_dir.exists():
            return
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for cache_file in self.cache_dir.glob("*"):
            if cache_file.is_file():
                # Check file age
                file_age = current_time - cache_file.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        cache_file.unlink()
                        print(f"Cleaned up old cache file: {cache_file.name}")
                    except Exception as e:
                        print(f"Warning: Could not delete cache file {cache_file.name}: {e}")
    
    
    async def _upload_file(
        self,
        file_path: Union[str, Path]
    ) -> File:
        """Upload a file to Google GenAI and return the file URI"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check if file already exists on Google AI
        existing_uri = await self._find_existing_file(file_path)
        if existing_uri:
            return existing_uri
        
        
        # Upload the file
        upload_response = await self.client.aio.files.upload(
            file=str(file_path)
        )
        
        return upload_response  # Return full file object for direct use in content
    
    def _upload_file_sync(
        self,
        file_path: Union[str, Path]
    ) -> File:
        """Upload a file to Google GenAI and return the file URI (sync)"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check if file already exists on Google AI
        existing_uri = self._find_existing_file_sync(file_path)
        if existing_uri:
            return existing_uri
        
        
        # Upload the file
        upload_response = self.client.files.upload(
            file=str(file_path)
        )
        
        return upload_response  # Return full file object for direct use in content
    
    async def _download_url_content(self, url: str) -> Tuple[bytes, str, str]:
        """Download content from URL and determine appropriate file extension and MIME type"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
                
                content = response.content
                content_type = response.headers.get('content-type', '').lower()
                
                # Determine file extension based on content type or URL
                if 'pdf' in content_type:
                    extension = '.pdf'
                    mime_type = 'application/pdf'
                elif 'html' in content_type or 'text/html' in content_type:
                    extension = '.html'
                    mime_type = 'text/html'
                elif 'json' in content_type:
                    extension = '.json'
                    mime_type = 'application/json'
                elif 'xml' in content_type:
                    extension = '.xml'
                    mime_type = 'application/xml'
                elif 'csv' in content_type:
                    extension = '.csv'
                    mime_type = 'text/csv'
                elif 'plain' in content_type or 'text/plain' in content_type:
                    extension = '.txt'
                    mime_type = 'text/plain'
                else:
                    # Try to infer from URL path
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    path = parsed.path.lower()
                    
                    if path.endswith('.pdf'):
                        extension = '.pdf'
                        mime_type = 'application/pdf'
                    elif path.endswith(('.html', '.htm')):
                        extension = '.html'
                        mime_type = 'text/html'
                    elif path.endswith('.json'):
                        extension = '.json'
                        mime_type = 'application/json'
                    elif path.endswith('.xml'):
                        extension = '.xml'
                        mime_type = 'application/xml'
                    elif path.endswith('.csv'):
                        extension = '.csv'
                        mime_type = 'text/csv'
                    elif path.endswith('.txt'):
                        extension = '.txt'
                        mime_type = 'text/plain'
                    else:
                        # Default to text for web content
                        extension = '.txt'
                        mime_type = 'text/plain'
                
                return content, extension, mime_type
                
        except Exception as e:
            raise Exception(f"Failed to download content from {url}: {str(e)}")
    
    def _download_url_content_sync(self, url: str) -> Tuple[bytes, str, str]:
        """Download content from URL and determine appropriate file extension and MIME type (sync)"""
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(url, follow_redirects=True)
                response.raise_for_status()
                
                content = response.content
                content_type = response.headers.get('content-type', '').lower()
                
                # Determine file extension based on content type or URL
                if 'pdf' in content_type:
                    extension = '.pdf'
                    mime_type = 'application/pdf'
                elif 'html' in content_type or 'text/html' in content_type:
                    extension = '.html'
                    mime_type = 'text/html'
                elif 'json' in content_type:
                    extension = '.json'
                    mime_type = 'application/json'
                elif 'xml' in content_type:
                    extension = '.xml'
                    mime_type = 'application/xml'
                elif 'csv' in content_type:
                    extension = '.csv'
                    mime_type = 'text/csv'
                elif 'plain' in content_type or 'text/plain' in content_type:
                    extension = '.txt'
                    mime_type = 'text/plain'
                else:
                    # Try to infer from URL path
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    path = parsed.path.lower()
                    
                    if path.endswith('.pdf'):
                        extension = '.pdf'
                        mime_type = 'application/pdf'
                    elif path.endswith(('.html', '.htm')):
                        extension = '.html'
                        mime_type = 'text/html'
                    elif path.endswith('.json'):
                        extension = '.json'
                        mime_type = 'application/json'
                    elif path.endswith('.xml'):
                        extension = '.xml'
                        mime_type = 'application/xml'
                    elif path.endswith('.csv'):
                        extension = '.csv'
                        mime_type = 'text/csv'
                    elif path.endswith('.txt'):
                        extension = '.txt'
                        mime_type = 'text/plain'
                    else:
                        # Default to text for web content
                        extension = '.txt'
                        mime_type = 'text/plain'
                
                return content, extension, mime_type
                
        except Exception as e:
            raise Exception(f"Failed to download content from {url}: {str(e)}")
    
    async def _upload_content_directly(self, content: bytes, url: str, content_type: Optional[str] = None) -> File:
        """Upload content directly to Google GenAI using cache directory"""
        try:
            # Determine file extension
            extension = self._get_file_extension(url, content_type)
            
            # Create cache filename
            cache_filename = self._create_cache_filename(url, content, extension)
            cache_file_path = self.cache_dir / cache_filename
            
            # Write content to cache file
            with open(cache_file_path, 'wb') as cache_file:
                cache_file.write(content)
            
            try:
                # Upload using the existing method
                upload_response = await self.client.aio.files.upload(
                    file=str(cache_file_path)
                )
                
                return upload_response  # Return full file object for direct use in content
            finally:
                # Clean up cache file
                if cache_file_path.exists():
                    cache_file_path.unlink()
                    
        except Exception as e:
            raise Exception(f"Failed to upload content to Gemini: {str(e)}")
    
    def _upload_content_directly_sync(self, content: bytes, url: str, content_type: Optional[str] = None) -> File:
        """Upload content directly to Google GenAI using cache directory (sync)"""
        try:
            # Determine file extension
            extension = self._get_file_extension(url, content_type)
            
            # Create cache filename
            cache_filename = self._create_cache_filename(url, content, extension)
            cache_file_path = self.cache_dir / cache_filename
            
            # Write content to cache file
            with open(cache_file_path, 'wb') as cache_file:
                cache_file.write(content)
            
            try:
                # Upload using the existing method
                upload_response = self.client.files.upload(
                    file=str(cache_file_path)
                )
                
                return upload_response  # Return full file object for direct use in content
            finally:
                # Clean up cache file
                if cache_file_path.exists():
                    cache_file_path.unlink()
                    
        except Exception as e:
            raise Exception(f"Failed to upload content to Gemini: {str(e)}")
    
    async def download_and_upload_url(self, url: str) -> File:
        """Download content from URL and upload directly to Gemini"""
        # Generate a filename based on URL
        from urllib.parse import urlparse
        parsed = urlparse(url)
        filename = parsed.path.split('/')[-1] if parsed.path.split('/')[-1] else f"url_content_{hash(url)}"
        
        # Download content
        content, extension, mime_type = await self._download_url_content(url)
        
        # Ensure filename has proper extension
        if not filename.endswith(extension):
            filename = f"{filename}{extension}"
        
        # Upload to Gemini
        return await self._upload_content_directly(content, url, mime_type)
    
    def download_and_upload_url_sync(self, url: str) -> File:
        """Download content from URL and upload directly to Gemini (sync)"""
        # Generate a filename based on URL
        from urllib.parse import urlparse
        parsed = urlparse(url)
        filename = parsed.path.split('/')[-1] if parsed.path.split('/')[-1] else f"url_content_{hash(url)}"
        
        # Download content
        content, extension, mime_type = self._download_url_content_sync(url)
        
        # Ensure filename has proper extension
        if not filename.endswith(extension):
            filename = f"{filename}{extension}"
        
        # Upload to Gemini
        return self._upload_content_directly_sync(content, url, mime_type)
    
    async def _prepare_content_with_files(
        self,
        prompt: str,
        file_uris: Optional[List[str]] = None,
        file_paths: Optional[List[Union[str, Path]]] = None
    ) -> List[Any]:
        """Prepare content parts including files"""
        content_parts = [prompt]
        
        # Upload files if file_paths provided
        if file_paths:
            file_uris = file_uris or []
            for file_path in file_paths:
                file_uri = await self._upload_file(file_path)
                file_uris.append(file_uri)
        
        # Add file objects to content
        if file_uris:  # Note: despite the name, these are now file objects
            for file_obj in file_uris:
                # Add file object directly - GenAI accepts file objects in content
                content_parts.append(file_obj)
        
        return content_parts
    
    def _prepare_content_with_files_sync(
        self,
        prompt: str,
        file_uris: Optional[List[str]] = None,
        file_paths: Optional[List[Union[str, Path]]] = None
    ) -> List[Any]:
        """Prepare content parts including files (sync)"""
        content_parts = [prompt]
        
        # Upload files if file_paths provided
        if file_paths:
            file_uris = file_uris or []
            for file_path in file_paths:
                file_uri = self._upload_file_sync(file_path)
                file_uris.append(file_uri)
        
        # Add file objects to content
        if file_uris:  # Note: despite the name, these are now file objects
            for file_obj in file_uris:
                # Add file object directly - GenAI accepts file objects in content
                content_parts.append(file_obj)
        
        return content_parts
    
    def _build_config(
        self, 
        system_prompt: Optional[str] = None,
        tools: Optional[List] = None,
        temperature: float = 0.0,
        response_mime_type: Optional[str] = None,
        response_schema: Optional[Dict] = None,
        **config_kwargs
    ) -> GenerateContentConfig:
        """Build config for content generation"""
        config_params = {
            "response_modalities": ["TEXT"],
            "temperature": temperature,
            "candidate_count": 1,
            **config_kwargs
        }
        
        if system_prompt:
            config_params["system_instruction"] = system_prompt
        if tools:
            config_params["tools"] = tools
        if response_mime_type:
            config_params["response_mime_type"] = response_mime_type
        if response_schema:
            config_params["response_schema"] = response_schema
            
        return GenerateContentConfig(**config_params)

    async def generate_content(
        self, 
        prompt: str,
        model_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List] = None,
        temperature: float = 0.0,
        response_mime_type: Optional[str] = None,
        response_schema: Optional[Dict] = None,
        file_uris: Optional[List[str]] = None,
        file_paths: Optional[List[Union[str, Path]]] = None,
        **config_kwargs
    ) -> Any:
        """
        Generate content using Gemini API (async)
        
        Args:
            prompt: User prompt/content to send
            model_id: Gemini model to use (overrides default if provided)
            system_prompt: Optional system instruction
            tools: Optional tools to include (e.g., search_tool)
            temperature: Generation temperature
            response_mime_type: MIME type for structured output
            response_schema: Schema for structured output
            file_uris: List of already-uploaded file URIs
            file_paths: List of local file paths to upload and include
            **config_kwargs: Additional config parameters
            
        Returns:
            Response object from Gemini API
        """
        model_to_use = model_id or self.default_model_id
        config = self._build_config(
            system_prompt=system_prompt,
            tools=tools,
            temperature=temperature,
            response_mime_type=response_mime_type,
            response_schema=response_schema,
            **config_kwargs
        )

        # Prepare content with optional files
        content_parts = await self._prepare_content_with_files(prompt, file_uris, file_paths)

        response = await self.client.aio.models.generate_content(
            model=model_to_use,
            contents=content_parts,
            config=config
        )
        
        return response

    def generate_content_sync(
        self, 
        prompt: str,
        model_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List] = None,
        temperature: float = 0.0,
        response_mime_type: Optional[str] = None,
        response_schema: Optional[Dict] = None,
        file_uris: Optional[List[str]] = None,
        file_paths: Optional[List[Union[str, Path]]] = None,
        **config_kwargs
    ) -> Any:
        """
        Generate content using Gemini API (sync)
        
        Args:
            prompt: User prompt/content to send
            model_id: Gemini model to use (overrides default if provided)
            system_prompt: Optional system instruction
            tools: Optional tools to include (e.g., search_tool)
            temperature: Generation temperature
            response_mime_type: MIME type for structured output
            response_schema: Schema for structured output
            file_uris: List of already-uploaded file URIs
            file_paths: List of local file paths to upload and include
            **config_kwargs: Additional config parameters
            
        Returns:
            Response object from Gemini API
        """
        model_to_use = model_id or self.default_model_id
        config = self._build_config(
            system_prompt=system_prompt,
            tools=tools,
            temperature=temperature,
            response_mime_type=response_mime_type,
            response_schema=response_schema,
            **config_kwargs
        )

        # Prepare content with optional files
        content_parts = self._prepare_content_with_files_sync(prompt, file_uris, file_paths)

        response = self.client.models.generate_content(
            model=model_to_use,
            contents=content_parts,
            config=config
        )
        
        return response
    
    async def search(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List] = None,
        include_google_search: bool = False,
        include_url_context: bool = True,
        file_uris: Optional[List[str]] = None,
        file_paths: Optional[List[Union[str, Path]]] = None,
        **config_kwargs
    ) -> Dict[str, Any]:
        """
        Generate content and automatically extract citations
        
        Args:
            prompt: User prompt
            model_id: Gemini model to use (overrides default if provided)
            system_prompt: Optional system instruction  
            tools: Optional tools (e.g., search_tool)
            include_google_search: Whether to include Google search tool
            include_url_context: Whether to include URL context tool
            file_uris: List of already-uploaded file URIs
            file_paths: List of local file paths to upload and include
            **config_kwargs: Additional config parameters
            
        Returns:
            Dictionary with response text, citations, and reference links
        """
        if include_google_search:
            tools = tools or []
            tools.append(Tool(google_search=GoogleSearch()))

        if include_url_context:
            tools = tools or []
            tools.append(Tool(url_context=UrlContext))

        response = await self.generate_content(
            prompt=prompt,
            model_id=model_id,
            system_prompt=system_prompt,
            tools=tools,
            file_uris=file_uris,
            file_paths=file_paths,
            **config_kwargs
        )
        
        citations = extract_citations(response)
        original_text = response.text
        text_with_citations = add_inline_citations(response.text, citations)
        
        result = {
            'text': original_text,
            'citations': citations,
            'text_with_citations': text_with_citations
        }
            
        return result

    def search_sync(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List] = None,
        include_google_search: bool = False,
        include_url_context: bool = True,
        file_uris: Optional[List[str]] = None,
        file_paths: Optional[List[Union[str, Path]]] = None,
        **config_kwargs
    ) -> Dict[str, Any]:
        """
        Generate content and automatically extract citations (sync)
        
        Args:
            prompt: User prompt
            model_id: Gemini model to use (overrides default if provided)
            system_prompt: Optional system instruction  
            tools: Optional tools (e.g., search_tool)
            include_google_search: Whether to include Google search tool
            include_url_context: Whether to include URL context tool
            file_uris: List of already-uploaded file URIs
            file_paths: List of local file paths to upload and include
            **config_kwargs: Additional config parameters
            

        Returns:
            Dictionary with response text, citations, and reference links
        """
        if include_google_search:
            tools = tools or []
            tools.append(Tool(google_search=GoogleSearch()))

        if include_url_context:
            tools = tools or []
            tools.append(Tool(url_context=UrlContext))

        response = self.generate_content_sync(
            prompt=prompt,
            model_id=model_id,
            system_prompt=system_prompt,
            tools=tools,
            file_uris=file_uris,
            file_paths=file_paths,
            **config_kwargs
        )
        
        citations = extract_citations(response)
        original_text = response.text
        
        result = {
            'text': original_text,
            'citations': citations,
        }
            
        return result
    
    async def validate_search(
        self,
        citations: List['Citation'],
        model_id: Optional[str] = None,
        tools: Optional[List] = None,
        include_google_search: bool = False,
        include_url_context: bool = True,
        file_uris: Optional[List[str]] = None,
        file_paths: Optional[List[Union[str, Path]]] = None,
        **config_kwargs
    ) -> str:

        if include_google_search:
            tools = tools or []
            tools.append(Tool(google_search=GoogleSearch()))

        if include_url_context:
            tools = tools or []
            tools.append(Tool(url_context=UrlContext))

        response = await self.generate_content(
            prompt=self.create_user_prompt_with_citations(citations=citations),
            model_id=model_id,
            system_prompt=self.create_citation_validation_system_prompt(),
            tools=tools,
            file_uris=file_uris,
            file_paths=file_paths,
            **config_kwargs
        )

        json_response = to_json_string(response.text)
        return json_response
    
    async def search_with_url_context(
        self,
        citations: List['Citation'],
        prompt: str,
        system_prompt: Optional[str] = None,
        model_id: Optional[str] = None,
        tools: Optional[List] = None,
        include_inline_citations: bool = False,
        placeholder_part: str = "S",
        file_uris: Optional[List[str]] = None,
        file_paths: Optional[List[Union[str, Path]]] = None,
        **config_kwargs
    ) -> str:

        tools = tools or []
        tools.append(Tool(url_context=UrlContext))
        
        urls = list({seg.original_link for c in citations for seg in c.segments if seg.original_link})
        if urls:
            prompt += "\n\nUse the following URLs as context:\n" + "\n".join(urls)

        response = await self.generate_content(
            prompt=prompt,
            model_id=model_id,
            system_prompt=system_prompt,
            tools=tools,
            file_uris=file_uris,
            file_paths=file_paths,
            **config_kwargs
        )

        filtered_citations = await update_citations(extract_citations(response), placeholder_part=placeholder_part, resolve=False)
 
        result = {
            'original_text': response.text,
            'citations': filtered_citations,
            'response_object': response,
            'text_with_citations': None,
            "removable_citations": []
        }
        
        if include_inline_citations:
            text_with_citations = add_inline_citations(response.text, filtered_citations)
            result['text_with_citations'] = text_with_citations
        
        return result
    
    async def generate_structured_output(
        self,
        content: str,
        model_id: Optional[str] = None,
        response_mime_type: Optional[str] = 'application/json',
        response_schema: Optional[Dict] = None,
        url_map: dict[str, str] | None = None,
        file_uris: Optional[List[str]] = None,
        file_paths: Optional[List[Union[str, Path]]] = None,
        **config_kwargs
    ) -> str:
        """
        Generate structured output from content
        
        Args:
            content: Content to process
            model_id: Gemini model to use (overrides default if provided)
            response_mime_type: MIME type for structured output
            response_schema: Schema for structured output
            url_map: Optional URL mapping for unmasking
            file_uris: List of already-uploaded file URIs
            file_paths: List of local file paths to upload and include
            **config_kwargs: Additional config parameters
            
        Returns:
            Structured output as string
        """
        config_kwargs["thinking_config"] = ThinkingConfig(include_thoughts=True, thinking_budget=1024)

        response = await self.generate_content(
            prompt=self.create_user_prompt_with_content(content),
            model_id=model_id,
            system_prompt=self.create_citation_enhanced_system_prompt(),
            response_mime_type=response_mime_type,
            response_schema=response_schema,
            file_uris=file_uris,
            file_paths=file_paths,
            **config_kwargs
        )

        structured = response.text
        if url_map:
            structured = unmask_urls(structured, url_map)
        return structured

    # Utility methods for file management
    async def list_uploaded_files(self) -> List[Dict]:
        """List all files currently stored on Google AI"""
        try:
            files_response = await self.client.aio.files.list()
            return [
                {
                    'uri': file_info.uri,
                    'display_name': file_info.display_name,
                    'mime_type': file_info.mime_type,
                    'size_bytes': getattr(file_info, 'size_bytes', None),
                    'create_time': getattr(file_info, 'create_time', None),
                    'update_time': getattr(file_info, 'update_time', None),
                    'state': getattr(file_info, 'state', None)
                }
                for file_info in files_response.files
            ]
        except Exception as e:
            print(f"Error listing files: {e}")
            return []

    def list_uploaded_files_sync(self) -> List[Dict]:
        """List all files currently stored on Google AI (sync)"""
        try:
            files_response = self.client.files.list()
            return [
                {
                    'uri': file_info.uri,
                    'display_name': file_info.display_name,
                    'mime_type': file_info.mime_type,
                    'size_bytes': getattr(file_info, 'size_bytes', None),
                    'create_time': getattr(file_info, 'create_time', None),
                    'update_time': getattr(file_info, 'update_time', None),
                    'state': getattr(file_info, 'state', None)
                }
                for file_info in files_response.files
            ]
        except Exception as e:
            print(f"Error listing files: {e}")
            return []

    async def delete_file(self, file_uri: str) -> bool:
        """Delete an uploaded file from Google's servers"""
        try:
            await self.client.aio.files.delete(name=file_uri)
            return True
        except Exception as e:
            print(f"Error deleting file {file_uri}: {e}")
            return False

    def delete_file_sync(self, file_uri: str) -> bool:
        """Delete an uploaded file from Google's servers (sync)"""
        try:
            self.client.files.delete(name=file_uri)
            return True
        except Exception as e:
            print(f"Error deleting file {file_uri}: {e}")
            return False

    async def get_file_info(self, file_uri: str) -> Optional[Dict]:
        """Get information about an uploaded file"""
        try:
            file_info = await self.client.aio.files.get(name=file_uri)
            return {
                'uri': file_info.uri,
                'display_name': file_info.display_name,
                'mime_type': file_info.mime_type,
                'size_bytes': getattr(file_info, 'size_bytes', None),
                'create_time': getattr(file_info, 'create_time', None),
                'update_time': getattr(file_info, 'update_time', None),
                'state': getattr(file_info, 'state', None)
            }
        except Exception as e:
            print(f"Error getting file info for {file_uri}: {e}")
            return None

    def get_file_info_sync(self, file_uri: str) -> Optional[Dict]:
        """Get information about an uploaded file (sync)"""
        try:
            file_info = self.client.files.get(name=file_uri)
            return {
                'uri': file_info.uri,
                'display_name': file_info.display_name,
                'mime_type': file_info.mime_type,
                'size_bytes': getattr(file_info, 'size_bytes', None),
                'create_time': getattr(file_info, 'create_time', None),
                'update_time': getattr(file_info, 'update_time', None),
                'state': getattr(file_info, 'state', None)
            }
        except Exception as e:
            print(f"Error getting file info for {file_uri}: {e}")
            return None

    async def find_files_by_name(self, name: str) -> List[Dict]:
        """Find files by display name"""
        all_files = await self.list_uploaded_files()
        return [f for f in all_files if f['display_name'] == name]

    def find_files_by_name_sync(self, name: str) -> List[Dict]:
        """Find files by display name (sync)"""
        all_files = self.list_uploaded_files_sync()
        return [f for f in all_files if f['display_name'] == name]

    def create_citation_validation_system_prompt(self) -> str:
        return textwrap.dedent("""
            You are a factual validation agent. Your task is to evaluate a list of citations. Each citation represents a document span (claim) and contains one or more supporting segments linked to URLs.

            Each citation has:
            - `text`: the referenced segment from the document (claim to validate)
            - `segments`: a list of associated web sources that may support the claim

            Each segment includes:
            - `chunk_index`: index in the source list
            - `link`: intermediate or redirected URL
            - `original_link`: the resolved final destination URL
            - `placeholder`: an identifier used in the text for this segment

            Your task:
            - For each segment in each citation, determine whether the content at `original_link` supports the claim (`text`)
            - Provide a relevance label: `"relevant"`, `"partial"`, or `"irrelevant"`
            - Assign a confidence score between 0 and 1
            - Provide a brief `evidence_snippet`: either a direct quote or paraphrased support from the page content

            Return only valid JSON in the format:
            {
            "verdicts": [
                {
                "citation_text": string,
                "chunk_index": int,
                "original_link": string,
                "placeholder": string,
                "relevance": "relevant" | "partial" | "irrelevant",
                "confidence": float,
                "evidence_snippet": string
                }
            ]
            }

            Notes:
            - Each citation can have multiple segments — evaluate each segment independently.
            - Include all segment evaluations in the output.
            - Do not include explanations, comments, or markdown — return only valid JSON.
        """)

    def create_user_prompt_with_citations(self, citations: list) -> str:
        """
        Builds a user prompt using serialized Citation objects based on the updated model.

        Each Citation includes:
        - `text`: the document segment or claim
        - `segments`: a list of web-based sources that may support the text
            - Each segment has a `chunk_index`, `link`, `original_link`, and `placeholder`

        Your job is to assess whether each segment supports the claim.

        Returns:
            A complete prompt string for Gemini validation.
        """
        serialized = json.dumps([c.dict() for c in citations], indent=2)

        return textwrap.dedent(f"""
            You are given citation metadata used to validate factual claims in a document.

            Each citation includes:
            - `text`: A specific segment (claim) from the document.
            - `segments`: A list of sources potentially supporting the text.
                - Each segment has:
                    - `chunk_index`: numeric index pointing to the chunk
                    - `link`: intermediate or redirected URL
                    - `original_link`: final destination URL
                    - `placeholder`: identifier token used in the original text

            Task:
            - For each segment, determine if the `original_link` supports the claim (`text`)
            - Assess relevance as one of: "relevant", "partial", or "irrelevant"
            - Provide a confidence score between 0 and 1
            - Include a short `evidence_snippet` from the content to justify the verdict

            Return only structured JSON using the format defined in the system prompt.

            citation_metadata:
            {serialized}
        """)

# utility methods

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




def unmask_urls(text: str, url_map: Dict[str, str]) -> str:
    for ph in sorted(url_map, key=len, reverse=True):
        text = text.replace(ph, url_map[ph])
    return text


def extract_citations(response) -> List[Citation]:
    """
    Extract citations from Gemini response grounding metadata
    
    Args:
        response: Gemini API response object
        
    Returns:
        List of Citation objects
    """
    citations = []
    
    try:
        if not hasattr(response, 'candidates') or not response.candidates:
            return citations
            
        candidate = response.candidates[0]
        if not hasattr(candidate, 'grounding_metadata') or not candidate.grounding_metadata:
            return citations
            
        grounding_metadata = candidate.grounding_metadata
        if not hasattr(grounding_metadata, 'grounding_supports'):
            return citations
            
        for support in grounding_metadata.grounding_supports or []:
            try:
                if not hasattr(support, "segment") or support.segment is None:
                    continue  # Skip this support if segment info is missing  
                
                start_index = (
                    support.segment.start_index
                    if support.segment.start_index is not None
                    else 0
                )

                # Ensure end_index is present to form a valid segment
                if support.segment.end_index is None:
                    continue  # Skip if end_index is missing, as it's crucial

                end_index = support.segment.end_index        
                confidence_scores = getattr(support, 'confidence_scores', [])
                segments = []
                citation = Citation(
                        score=float(confidence_scores[0] if confidence_scores else 0.0),
                        start_index=int(start_index),
                        end_index=int(end_index),
                        text=support.segment.text if hasattr(support.segment, 'text') else None,
                        segments=segments
                    )
                citations.append(citation)
                
                if hasattr(support, "grounding_chunk_indices") and support.grounding_chunk_indices:
                    for chunk_index in support.grounding_chunk_indices:
                        try:
                            chunk = candidate.grounding_metadata.grounding_chunks[chunk_index]
                            segment = Segments(
                                title=(chunk.web.title.split(".")[:-1][0] if chunk.web and chunk.web.title and "." in chunk.web.title else ""),
                                chunk_index=int(chunk_index),
                                link=str(getattr(chunk.web, 'uri', '') or '')
                            )
                            segments.append(segment)    
                            
                        except (AttributeError, IndexError, TypeError, ValueError):
                            # Handle cases where chunk, web, uri, or resolved_map might be problematic
                            # For simplicity, we'll just skip adding this particular segment link
                            # In a production system, you might want to log this.
                            pass
                    citation.segments = segments    

            except (AttributeError, IndexError, TypeError, ValueError):
                continue
    except Exception:
        return citations
    
    return citations
    

def add_inline_citations(text: str, citations: List[Citation]):
    """
    Inserts citation markers into a text string based on start and end indices.

    Args:
        text (str): The original text string.
        citations_list (list): A list of dictionaries, where each dictionary
                            contains 'start_index', 'end_index', and
                            'segment_string' (the marker to insert).
                            Indices are assumed to be for the original text.

    Returns:
        str: The text with citation markers inserted.
    """
    # Sort citations by end_index in descending order.
    # If end_index is the same, secondary sort by start_index descending.
    # This ensures that insertions at the end of the string don't affect
    # the indices of earlier parts of the string that still need to be processed.
    sorted_citations = sorted(
        citations, key=lambda c: (c.end_index, c.start_index), reverse=True
    )

    modified_text = text
    text_encoded = modified_text.encode("utf-8")

    for citation_info in sorted_citations:
        # These indices refer to positions in the *original* text,
        # but since we iterate from the end, they remain valid for insertion
        # relative to the parts of the string already processed.
        end_idx = citation_info.end_index

        for segment in citation_info.segments:

            marker_to_insert = ""
            marker_to_insert += segment.get_formated_link()
            # Insert the citation marker at the original end_idx position
            char_end_index =  len(text_encoded[:end_idx].decode("utf-8", errors="ignore"))

            modified_text = (
                modified_text[:char_end_index] + marker_to_insert + modified_text[char_end_index:]
            )

    return modified_text


def filter_citations_by_verdicts(
    verdict_payload_json: str,
    citations: List[Citation],
    removable_citations: List[Citation]
) -> Tuple[bool, List[Citation], List[Citation]]:
    """
    Filters out segments marked as irrelevant and optionally removes segments by domain.
    If all segments in a citation are removed, the citation text is also removed from content.

    Args:
        citations: List of Citation objects
        verdict_payload_json: JSON string with verdicts from Gemini
        content: The full content string
        my_domain: Optional domain to exclude (e.g., 'example.com')

    Returns:
        (is_valid: bool, has_relevant: bool, updated_citations: List[Citation], updated_content: str)
    """

    if not verdict_payload_json:
        return False, [], citations

    try:
        verdict_payload = json.loads(verdict_payload_json)
    except json.JSONDecodeError:
        return False, [], citations
    
    verdicts = verdict_payload.get("verdicts", [])
    if not verdicts:
        return False, [], citations

    relevance_map: Dict[str, str] = {
        v.get("placeholder"): v.get("relevance", "irrelevant")
        for v in verdicts
    }

    updated_citations: List[Citation] = []

    for cit in citations:
        # Step 1: Filter by verdicts
        filtered_segments = [
            seg for seg in cit.segments
            if relevance_map.get(seg.placeholder) == "relevant"
        ]

        if not filtered_segments:
            removable_citations.append(cit)
            continue  # Skip adding this citation

        # Step 2: Keep citation with remaining segments
        cit.segments = filtered_segments
        updated_citations.append(cit)

    return True, updated_citations, removable_citations


def group_citations_by_original_link(citations: List[Citation]) -> Dict[Optional[str], List[Citation]]:
    """
    Group citations by original_link. For each URL group, citations only contain 
    segments that match that specific original_link.
    
    Args:
        citations: List of Citation objects
        
    Returns:
        Dictionary mapping original_link -> List of citations with only matching segments
    """
    groups = defaultdict(list)
    
    for citation in citations:
        # Group segments by their original_link
        segments_by_link = defaultdict(list)
        for segment in citation.segments:
            segments_by_link[segment.original_link].append(segment)
        
        # For each original_link, create a citation copy with only matching segments
        for original_link, matching_segments in segments_by_link.items():
            citation_copy = Citation(
                score=citation.score,
                start_index=citation.start_index,
                end_index=citation.end_index,
                text=citation.text,
                segments=matching_segments
            )
            groups[original_link].append(citation_copy)
    
    return dict(groups)

def remove_content_by_citations(citations: List[Citation], content: str) -> str:
    for cit in citations:
        # Remove the text segment from the content
        if cit.text:
            content = content.replace(cit.text, "")
    
    return content.strip()


def to_json_string(raw: Any) -> str:
        """Return a clean JSON string ready for JsonOutputParser."""

        FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```$", re.I | re.M)

        if isinstance(raw, list):
            raw = raw[0]

        # remove markdown code fences
        if isinstance(raw, str):
            raw = FENCE.sub("", raw).strip()

        # ensure final payload is string
        if not isinstance(raw, str):
            raw = json.dumps(raw, ensure_ascii=False)  

        return raw

async def update_citations(citations: List[Citation], placeholder_part:str = "T", resolve:bool= True) -> List[Citation]:
    """
    Resolve original_link for each segment inside each citation.
    Assigns a unique placeholder per segment.
    Updates each segment in-place.
    """

    async def process_segment(global_idx: int, segment: Segments):
        if segment.chunk_index >= 0:
            segment.placeholder = f"URL_{placeholder_part}_{global_idx}"
            segment.original_link = (await resolve_redirect(segment.link) or segment.link) if resolve else segment.link

    async def process_citation(start_idx: int, cit: Citation):
        await asyncio.gather(
            *[process_segment(start_idx + idx, seg) for idx, seg in enumerate(cit.segments)]
        )
        return cit

    # Flattened indexing to ensure global uniqueness across all segments
    idx_counter = 1
    tasks = []
    for cit in citations:
        tasks.append(process_citation(idx_counter, cit))
        idx_counter += len(cit.segments)

    await asyncio.gather(*tasks)
    return citations



async def resolve_redirect(redirect_url: str) -> str:
    try:
        REDIRECT_CODES = {301, 302, 303, 307, 308}

        async with httpx.AsyncClient() as client:
            response = await client.get(
                redirect_url,
                follow_redirects=False,
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            
            if response.status_code in REDIRECT_CODES and 'location' in response.headers:
                return response.headers['location']
            elif response.status_code == 200:
                return str(response.url)
            else:
                return redirect_url
    except httpx.RequestError as e:
        return redirect_url






# Example usage with Google AI file management
async def example_usage():
    """Examples of how to use the enhanced wrapper with Google AI file management"""
    
    gemini = GoogleGenAI(api_key="your-api-key")
    
    # Example 1: Generate content with file attachment (automatically checks for existing files)
    response = await gemini.generate_content(
        prompt="Analyze this document and provide key insights",
        file_paths=["report.pdf"],  # Will reuse if already uploaded
        temperature=0.1
    )
    
    # Example 2: List all files currently on Google AI
    files = await gemini.list_uploaded_files()
    print(f"Found {len(files)} files on Google AI:")
    for file in files:
        print(f"  - {file['display_name']} ({file['size_bytes']} bytes)")
    
    # Example 3: Find specific files by name
    existing_reports = await gemini.find_files_by_name("report.pdf")
    if existing_reports:
        print(f"Found existing report: {existing_reports[0]['uri']}")
        
        # Use existing file URI directly
        response = await gemini.generate_content(
            prompt="Summarize this report",
            file_uris=[existing_reports[0]['uri']]
        )
    
    # Example 4: Search with document context (reuses existing files)
    search_result = await gemini.search(
        prompt="What are the main findings about market trends?",
        file_paths=["market_analysis.xlsx", "survey_data.csv"],  # Checks Google AI first
        include_google_search=True,
        include_url_context=True
    )
    
    # Example 5: Clean up old files
    all_files = await gemini.list_uploaded_files()
    for file in all_files:
        if file['display_name'].startswith('temp_'):
            await gemini.delete_file(file['uri'])
            print(f"Deleted temporary file: {file['display_name']}")
    
    # Example 6: Structured output from documents
    structured = await gemini.generate_structured_output(
        content="Extract key metrics from these financial documents",
        file_paths=["q1_report.pdf", "q2_report.pdf"],  # Reuses if already uploaded
        response_mime_type="application/json",
        response_schema={
            "type": "object",
            "properties": {
                "metrics": {"type": "array", "items": {"type": "object"}},
                "summary": {"type": "string"}
            }
        }
    )
    
    return response, search_result, structured

# Helper function for batch file management
async def manage_document_library(gemini: GoogleGenAI, documents_dir: str):
    """Example of managing a document library with Google AI"""
    from pathlib import Path
    
    documents_dir = Path(documents_dir)
    local_files = list(documents_dir.glob("*.pdf"))
    
    print(f"Found {len(local_files)} local PDF files")
    
    # Check what's already uploaded
    uploaded_files = await gemini.list_uploaded_files()
    uploaded_names = {f['display_name'] for f in uploaded_files}
    
    # Upload only new files
    new_uploads = []
    for local_file in local_files:
        if local_file.name not in uploaded_names:
            print(f"Uploading new file: {local_file.name}")
            uri = await gemini._upload_file(local_file)
            new_uploads.append(uri)
        else:
            print(f"File already exists: {local_file.name}")
    
    print(f"Uploaded {len(new_uploads)} new files")
    
    # Now analyze all documents
    all_file_paths = [str(f) for f in local_files]
    analysis = await gemini.generate_content(
        prompt="Provide a comprehensive analysis of all these documents",
        file_paths=all_file_paths,  # Will reuse existing uploads
        temperature=0.1
    )
    
    return analysis