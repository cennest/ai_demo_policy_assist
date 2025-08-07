from collections import defaultdict
import json
import httpx
import json
import re
import asyncio
from typing import Any, Dict, Optional,List, Tuple
from citation import Citation, Segments


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


