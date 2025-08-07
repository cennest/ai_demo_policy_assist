import textwrap
from typing import List, Optional, Any, Dict
from google import genai
from google.genai.types import GenerateContentConfig,ThinkingConfig
from google.genai.types import Tool, GoogleSearch, UrlContext
from client_utils import *


class GoogleGenAI:
    """
    Handy wrapper for Google Gemini API operations including:
    - Content generation
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

        response = await self.client.aio.models.generate_content(
            model=model_to_use,
            contents=prompt,
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

        response = self.client.models.generate_content(
            model=model_to_use,
            contents=prompt,
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
        **config_kwargs
    ) -> Dict[str, Any]:
        """
        Generate content and automatically extract citations
        
        Args:
            prompt: User prompt
            model_id: Gemini model to use (overrides default if provided)
            system_prompt: Optional system instruction  
            tools: Optional tools (e.g., search_tool)
            include_inline_citations: Whether to add inline citations to text
            include_reference_links: Whether to include reference links
            include_google_search: Whether to include Google search tool
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
            **config_kwargs
        )
        

        citations = extract_citations(response)
        original_text = response.text
        text_with_citations = add_inline_citations(response.text, citations)
        
        result = {
            'text': original_text,
            'citations': citations,
            'text_with_citations' : text_with_citations
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
            **config_kwargs
        )
        

        citations = extract_citations(response)
        original_text = response.text
        #text_with_citations = add_inline_citations(response.text, citations)
        
        result = {
            'text': original_text,
            'citations': citations,
            #'text_with_citations' : "text_with_citations"
        }
            
        return result
    

    async def validate_search(
        self,
        citations: List[Citation],
        model_id: Optional[str] = None,
        tools: Optional[List] = None,
        include_google_search: bool = False,
        include_url_context: bool = True,
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
            **config_kwargs
        )

        json_response = to_json_string(response.text)
        return json_response
    

    async def search_with_url_context(
        self,
        citations: List[Citation],
        prompt: str,
        system_prompt: Optional[str] = None,
        model_id: Optional[str] = None,
        tools: Optional[List] = None,
        include_inline_citations: bool = False,
        placeholder_part:str="S",
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
            **config_kwargs
        )

        filtered_citations =  await update_citations(extract_citations(response), placeholder_part=placeholder_part,resolve=False)
 
        result = {
            'original_text': response.text,
            'citations': filtered_citations,
            'response_object': response,
            'text_with_citations' : None,
            "removable_citations" : []
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
        **config_kwargs
    ) -> str:
        """
        Generate structured output from content
        
        Args:
            content: Content to process
            model_id: Gemini model to use (overrides default if provided)
            response_mime_type: MIME type for structured output
            response_schema: Schema for structured output
            **config_kwargs: Additional config parameters
            
        Returns:
            Structured output as string
        """

        config_kwargs["thinking_config"] = ThinkingConfig(include_thoughts=True,thinking_budget=1024)

        response = await self.generate_content(
            prompt=self.create_user_prompt_with_content(content),
            model_id=model_id,
            system_prompt=self.create_citation_enhanced_system_prompt(),
            response_mime_type=response_mime_type,
            response_schema=response_schema,
            **config_kwargs
        )


        structured = response.text
        if url_map:
            structured = unmask_urls(structured, url_map)
        return structured



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