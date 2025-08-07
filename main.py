import time
import asyncio
import streamlit as st
from typing import List, Dict, Any, Optional
from google_gen_ai_client import GoogleGenAI
from config import ConfigManager

class PolicyChatApp:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        self.model: Optional[GoogleGenAI] = None
        self.chat_history: List[Dict[str, str]] = []
        self.last_error: Optional[str] = None
        self.setup_gemini()
    
    def setup_gemini(self) -> bool:
        """Initialize GoogleGenAI with custom client"""
        try:
            if not self.config.google_api_key or self.config.google_api_key == "<enter google api key>":
                self.last_error = "Google API key is not configured"
                self.model = None
                return False
            
            if not self.config.google_api_key.startswith("AIzaSy") or len(self.config.google_api_key) < 30:
                self.last_error = "Invalid Google API key format"
                self.model = None
                return False
            
            # Initialize custom GoogleGenAI client
            self.model = GoogleGenAI(
                api_key=self.config.google_api_key,
                default_model_id=self.config.gemini_model
            )
            
            self.last_error = None
            return True
            
        except Exception as e:
            self.last_error = f"Failed to initialize Gemini client: {str(e)}"
            self.model = None
            return False
    
    def is_ready(self) -> bool:
        """Check if the chat app is ready to process queries"""
        return self.model is not None and self.last_error is None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the chat app"""
        return {
            "ready": self.is_ready(),
            "model_initialized": self.model is not None,
            "api_key_valid": bool(
                self.config.google_api_key and 
                self.config.google_api_key != "<enter google api key>" and
                self.config.google_api_key.startswith("AIzaSy") and 
                len(self.config.google_api_key) > 30
            ),
            "last_error": self.last_error,
            "chat_history_count": len(self.chat_history),
            "policy_urls_count": len(self.config.policy_urls)
        }
    
    def refresh_config(self) -> None:
        """Refresh configuration and reinitialize if needed"""
        old_api_key = self.config.google_api_key
        old_model = self.config.gemini_model
        
        self.config = self.config_manager.get_config()
        
        # Reinitialize if key or model changed
        if (old_api_key != self.config.google_api_key or 
            old_model != self.config.gemini_model):
            self.setup_gemini()
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt with proper fallback"""
        if self.config.system_prompt and self.config.system_prompt.strip():
            return self.config.system_prompt.strip()
        
        return """Your job is to extract the most relevant information from the provided context URLs to answer user questions. Always use evidence-based answers from the given policy documents. Refer to chat history to maintain contextual awareness. When answering, use only information supported by the provided context URLs."""
    
    def _build_context_prompt(self, user_message: str) -> str:
        """Build the complete prompt with context and history"""
        prompt_parts = []
        
        # Add conversation history if enabled
        if self.config.include_chat_history and self.chat_history:
            max_messages = min(self.config.max_history_messages, len(self.chat_history))
            recent_messages = self.chat_history[-max_messages:]
            
            if recent_messages:
                prompt_parts.append("=== CONVERSATION HISTORY ===")
                for msg in recent_messages:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    # Truncate very long messages to keep context manageable
                    content = msg["content"][:1000] + "..." if len(msg["content"]) > 1000 else msg["content"]
                    prompt_parts.append(f"{role}: {content}")
                prompt_parts.append("")
        
        # Add URLs context if enabled
        if self.config.url_context_tool and self.config.policy_urls:
            prompt_parts.append("=== CONTEXT URLS ===")
            prompt_parts.append("Analyze the following URLs to answer the user's question:")
            for i, url in enumerate(self.config.policy_urls, 1):
                prompt_parts.append(f"{i}. {url}")
            prompt_parts.append("")
        
        # Add the current user question
        prompt_parts.append("=== CURRENT QUESTION ===")
        prompt_parts.append(user_message)
        
        return "\n".join(prompt_parts)
    
    def query_with_context(self, user_message: str) -> str:
        """Query Gemini with URL context and enhanced search capabilities for medical policy information"""
        try:
            # Check if model is ready
            if not self.is_ready():
                if self.last_error:
                    return f"❌ {self.last_error}"
                return "❌ Chat app is not properly initialized. Please check your configuration."
            
            # Validate input
            if not user_message or not user_message.strip():
                return "❌ Please provide a valid question."
            
            user_message = user_message.strip()
            
            # Build prompts
            system_prompt = self._build_system_prompt()
            full_prompt = self._build_context_prompt(user_message)
            
            # Log the query attempt (for debugging)
            st.write(f"🔍 Querying {self.config.gemini_model}...")
            
            # Query the model with timeout and retry logic
            start_time = time.time()
            
            try:
                result = self.model.search_sync(
                    prompt=full_prompt,
                    system_prompt=system_prompt,
                    temperature=0.1,
                    include_google_search=False,  # Focus on URL context for policies
                    include_url_context=self.config.url_context_tool
                )
                
                query_time = time.time() - start_time
                st.write(f"⏱️ Response generated in {query_time:.2f} seconds")
                
            except Exception as api_error:
                # Handle API-specific errors
                error_msg = str(api_error).lower()
                if "timeout" in error_msg:
                    return "⏱️ Request timed out. Please try again with a shorter question."
                elif "rate limit" in error_msg or "quota" in error_msg:
                    return "🚦 API rate limit exceeded. Please wait a moment and try again."
                elif "authentication" in error_msg or "api key" in error_msg:
                    return "🔑 Authentication error. Please check your API key configuration."
                elif "permission" in error_msg:
                    return "🚫 Permission denied. Please verify your API key has the necessary permissions."
                else:
                    raise api_error  # Re-raise for general error handling
            
            # Process the result
            if isinstance(result, dict):
                response_text = result.get('text_with_citations', result.get('text', ''))
                if not response_text:
                    return "❌ No response generated. Please try rephrasing your question."
                
                # Add metadata if available
                if 'sources' in result and result['sources']:
                    response_text += f"\n\n📋 **Sources consulted:** {len(result['sources'])} policy documents"
                
                return response_text
            
            elif isinstance(result, str):
                return result if result.strip() else "❌ Empty response received. Please try again."
            
            else:
                return str(result) if result else "❌ No response generated. Please try again."
            
        except Exception as e:
            error_msg = str(e)
            
            # Categorize errors for better user experience
            if any(keyword in error_msg.upper() for keyword in ["API_KEY", "AUTHENTICATION", "UNAUTHORIZED"]):
                return "🔑 API Key error. Please verify your Google API key is correct and has proper permissions."
            
            elif any(keyword in error_msg.upper() for keyword in ["QUOTA", "RATE_LIMIT", "TOO_MANY_REQUESTS"]):
                return "🚦 API quota exceeded or rate limited. Please wait and try again later."
            
            elif any(keyword in error_msg.upper() for keyword in ["PERMISSION", "FORBIDDEN", "ACCESS_DENIED"]):
                return "🚫 Permission denied. Please check your API key permissions for the Gemini API."
            
            elif any(keyword in error_msg.upper() for keyword in ["NETWORK", "CONNECTION", "TIMEOUT", "DNS"]):
                return "🌐 Network error. Please check your internet connection and try again."
            
            elif "MODEL_NOT_FOUND" in error_msg.upper():
                return f"🤖 Model '{self.config.gemini_model}' not found. Please check your model configuration."
            
            else:
                # Log the full error for debugging while showing a user-friendly message
                st.error(f"Debug info: {error_msg}")
                return "❌ An unexpected error occurred. Please try again or contact support if the issue persists."
    
    def add_to_memory(self, user_message: str, ai_response: str) -> None:
        """Add messages to conversation history with size management"""
        # Add new messages
        self.chat_history.append({"role": "user", "content": user_message})
        self.chat_history.append({"role": "assistant", "content": ai_response})
        
        # Trim history if it gets too long (keep last 100 messages max)
        max_total_messages = 100
        if len(self.chat_history) > max_total_messages:
            # Remove oldest messages but keep pairs (user + assistant)
            messages_to_remove = len(self.chat_history) - max_total_messages
            # Ensure we remove in pairs
            if messages_to_remove % 2 != 0:
                messages_to_remove += 1
            self.chat_history = self.chat_history[messages_to_remove:]
    
    def clear_memory(self) -> None:
        """Clear conversation history"""
        self.chat_history.clear()
    
    def get_memory_stats(self) -> Dict[str, int]:
        """Get statistics about conversation memory"""
        total_chars = sum(len(msg["content"]) for msg in self.chat_history)
        user_messages = len([msg for msg in self.chat_history if msg["role"] == "user"])
        assistant_messages = len([msg for msg in self.chat_history if msg["role"] == "assistant"])
        
        return {
            "total_messages": len(self.chat_history),
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "total_characters": total_chars,
            "average_message_length": total_chars // len(self.chat_history) if self.chat_history else 0
        }

def get_or_create_chat_app(config_manager: ConfigManager) -> PolicyChatApp:
    """Get existing chat app or create new one if config changed"""
    config = config_manager.get_config()
    
    # Calculate config hash for change detection
    config_hash = hash(str(sorted(config.__dict__.items())))
    
    # Check if we need to recreate the chat app
    should_recreate = (
        "chat_app" not in st.session_state or 
        st.session_state.get("last_api_key") != config.google_api_key or
        st.session_state.get("last_model") != config.gemini_model or
        st.session_state.get("last_config_hash") != config_hash
    )
    
    if should_recreate:
        # Create new chat app
        st.session_state.chat_app = PolicyChatApp(config_manager)
        st.session_state.last_api_key = config.google_api_key
        st.session_state.last_model = config.gemini_model
        st.session_state.last_config_hash = config_hash
        
        # Show status message
        status = st.session_state.chat_app.get_status()
        if status["ready"]:
            st.success("✅ Chat app initialized successfully!")
        else:
            st.warning(f"⚠️ Chat app created but not ready: {status['last_error']}")
    else:
        # Refresh existing app's configuration
        st.session_state.chat_app.refresh_config()
    
    return st.session_state.chat_app

def main():
    """Main application entry point with enhanced error handling and state management"""
    try:
        # Initialize config manager with error handling
        if "config_manager" not in st.session_state:
            try:
                st.session_state.config_manager = ConfigManager()
            except Exception as e:
                st.error(f"❌ Failed to initialize configuration: {str(e)}")
                st.stop()
        
        config_manager = st.session_state.config_manager
        config = config_manager.get_config()
        
        # Configure page with settings from config
        try:
            st.set_page_config(
                page_title=config.page_title,
                page_icon=config.page_icon,
                layout=config.layout,
                initial_sidebar_state=config.sidebar_state
            )
        except st.errors.StreamlitAPIException:
            # Page config already set, continue
            pass
        
        # Enhanced CSS with better responsive design and accessibility
        st.markdown(f"""
        <style>
        /* Responsive button styling */
        .stButton button {{
            font-size: 14px !important;
            padding: 0.25rem 0.5rem !important;
            height: 2.5rem !important;
            transition: all 0.2s ease !important;
        }}
        
        /* Delete button styling with hover effects */
        .stButton button[title*="Delete"] {{
            font-size: 16px !important;
            padding: 0.2rem !important;
            height: 2rem !important;
            background-color: #ff4b4b !important;
            color: white !important;
            border: none !important;
            border-radius: 4px !important;
        }}
        
        .stButton button[title*="Delete"]:hover {{
            background-color: #ff6b6b !important;
            transform: scale(1.05) !important;
        }}
        
        /* Sidebar responsive design */
        .css-1d391kg {{
            padding-top: 1rem;
            width: min(350px, 90vw) !important;
        }}
        
        /* Alert styling */
        .stAlert {{
            font-size: 14px;
            padding: 0.5rem 1rem;
            border-radius: 6px;
        }}
        
        /* Header hierarchy */
        h1 {{ font-size: 1.8rem !important; line-height: 1.2; }}
        h2 {{ font-size: 1.4rem !important; line-height: 1.2; }}
        h3 {{ font-size: 1.2rem !important; line-height: 1.2; }}
        
        /* Expandable sections */
        .stExpander {{ margin-bottom: 0.5rem; }}
        .stExpander summary {{ font-weight: 500; }}
        
        /* Input field styling */
        .stTextInput input, .stTextArea textarea {{
            padding: 0.5rem !important;
            font-size: 14px !important;
            border-radius: 6px !important;
        }}
        
        /* Chat input enhancement */
        .stChatInput input {{
            font-size: 16px !important;
            padding: 0.75rem !important;
        }}
        
        /* Theme-based background */
        .stApp {{
            background-color: {'#fafafa' if config.theme_base == 'light' else '#0e1117'};
        }}
        
        /* Loading spinner customization */
        .stSpinner {{ text-align: center; }}
        
        /* Error message styling */
        .error-container {{
            background-color: #ffebee;
            border-left: 4px solid #f44336;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 4px;
        }}
        
        /* Success message styling */
        .success-container {{
            background-color: #e8f5e8;
            border-left: 4px solid #4caf50;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 4px;
        }}
        
        /* Status indicator styling */
        .status-indicator {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            font-size: 0.875rem;
            font-weight: 500;
        }}
        
        .status-ready {{ background-color: #e8f5e8; color: #2e7d32; }}
        .status-error {{ background-color: #ffebee; color: #d32f2f; }}
        .status-warning {{ background-color: #fff3e0; color: #f57c00; }}
        [data-testid="stChatMessage"] > [data-testid="stChatMessageContent"] .stElementContainer[data-stale="true"] {{ display: none !important; }}
        </style>
        """, unsafe_allow_html=True)
        
        # Application header with status indicator
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title(f"{config.page_icon} {config.page_title}")
            st.markdown("*Ask questions about medical necessity criteria and policy requirements*")
        
        # Configuration validation
        config_errors = config_manager.validate_config()
        is_config_complete = len(config_errors) == 0
        
        # Status indicator
        with col2:
            if is_config_complete:
                st.markdown('<div class="status-indicator status-ready">✅ Ready</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-indicator status-error">⚠️ Setup Required</div>', unsafe_allow_html=True)
        
        # Configuration status banner
        if not is_config_complete:
            st.error("⚠️ **Configuration Required** - Please complete the settings in the sidebar before starting chat")
            with st.expander("❌ Configuration Issues", expanded=True):
                for field, error in config_errors.items():
                    st.markdown(f"• **{field}**: {error}")
            st.info("👈 Use the sidebar to configure your settings")
        
        # Initialize session state with defaults
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "chat_initialized" not in st.session_state:
            st.session_state.chat_initialized = False
        
        # Get or create chat app instance with enhanced error handling
        try:
            chat_app = get_or_create_chat_app(config_manager)
            
            # Validate chat app status
            status = chat_app.get_status()
            if not status["ready"] and is_config_complete:
                st.warning(f"⚠️ Chat app initialization issue: {status.get('last_error', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"❌ Failed to initialize chat application: {str(e)}")
            st.stop()
        
        # Enhanced sidebar with better organization
        with st.sidebar:
            # Status header
            if not is_config_complete:
                st.markdown("""
                <div class="error-container">
                    <h4>🔧 Setup Required</h4>
                    <p>Complete the following to start chatting:</p>
                </div>
                """, unsafe_allow_html=True)
                for field, error in config_errors.items():
                    st.markdown(f"• {error}")
                st.divider()
            else:
                st.markdown("""
                <div class="success-container">
                    <h4>✅ Configuration Complete</h4>
                    <p>Ready to analyze medical policies!</p>
                </div>
                """, unsafe_allow_html=True)
                st.divider()
            
            # Policy Configuration Section
            st.header("📋 Policy Configuration")
            
            # Add new policy URL with validation
            with st.form("add_url_form", clear_on_submit=True):
                new_url = st.text_input(
                    "Add Policy URL:", 
                    placeholder="https://example.com/policy",
                    help="Enter a valid URL to a medical policy document"
                )
                submitted = st.form_submit_button("Add URL", use_container_width=True)
                
                if submitted and new_url:
                    if new_url.startswith(('http://', 'https://')):
                        if config_manager.add_policy_url(new_url):
                            st.success(f"✅ Added: {new_url[:50]}...")
                            st.rerun()
                        else:
                            st.warning("⚠️ URL already exists")
                    else:
                        st.error("❌ Please enter a valid URL starting with http:// or https://")
            
            # Display current URLs with enhanced management
            st.subheader("Current Policy URLs:")
            if config.policy_urls:
                for i, url in enumerate(config.policy_urls):
                    with st.container():
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            # Show truncated URL with tooltip
                            display_url = url[:50] + "..." if len(url) > 50 else url
                            st.text(display_url)
                            if len(url) > 50:
                                st.caption(f"Full URL: {url}")
                        with col2:
                            if st.button("🗑️", key=f"remove_{i}", use_container_width=True, help=f"Delete {url}"):
                                if config_manager.remove_policy_url(url):
                                    st.success("URL removed")
                                    st.rerun()
            else:
                st.info("No policy URLs configured yet")
            
            st.divider()
            
            # Settings Section with tabs for better organization
            st.header("⚙️ Settings")
            
            # System Prompt Configuration
            with st.expander("💬 System Prompt", expanded=False):
                current_system_prompt = config.system_prompt if config.system_prompt else ""
                new_system_prompt = st.text_area(
                    "Custom System Prompt:",
                    value=current_system_prompt,
                    height=150,
                    placeholder="Enter custom system prompt (leave empty to use default)",
                    help="Customize how the AI responds. Leave empty to use the default medical policy assistant prompt."
                )
                if new_system_prompt != config.system_prompt:
                    config_manager.update_config({"system_prompt": new_system_prompt})
                    st.success("System prompt updated!")
            
            # Chat History Settings
            with st.expander("💾 Chat History", expanded=False):
                new_include_history = st.checkbox(
                    "Include Chat History in LLM Calls",
                    value=config.include_chat_history,
                    help="When enabled, conversation history is sent to the LLM for context."
                )
                if new_include_history != config.include_chat_history:
                    config_manager.update_config({"include_chat_history": new_include_history})
                
                if config.include_chat_history:
                    new_max_messages = st.slider(
                        "Max History Messages:", 
                        min_value=1, 
                        max_value=50, 
                        value=config.max_history_messages,
                        help="Maximum number of previous messages to include in context"
                    )
                    if new_max_messages != config.max_history_messages:
                        config_manager.update_config({"max_history_messages": new_max_messages})
            
            # Model and API Settings
            with st.expander("🤖 Model Settings", expanded=False):
                # Gemini model selection
                model_options = ["gemini-2.5-flash", "gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"]
                try:
                    current_model_idx = model_options.index(config.gemini_model)
                except ValueError:
                    current_model_idx = 0
                
                new_model = st.selectbox(
                    "Gemini Model:", 
                    options=model_options,
                    index=current_model_idx,
                    help="Choose the Gemini model for processing"
                )

                if new_model != config.gemini_model:
                    config_manager.update_config({"gemini_model": new_model})
                    # Use get_or_create_chat_app to refresh the chat app
                    chat_app = get_or_create_chat_app(config_manager)
                    st.success(f"Model changed to {new_model}")
                    st.rerun()
                
                # URL Context Tool
                new_url_context = st.checkbox(
                    "Enable URL Context Tool",
                    value=config.url_context_tool,
                    help="Use Gemini's URL context feature to analyze policy URLs directly"
                )
                if new_url_context != config.url_context_tool:
                    config_manager.update_config({"url_context_tool": new_url_context})
            
            # UI Settings
            with st.expander("🎨 UI Settings", expanded=False):
                # Theme selection
                theme_options = ["light", "dark"]
                try:
                    current_theme_idx = theme_options.index(config.theme_base)
                except ValueError:
                    current_theme_idx = 0
                
                new_theme = st.selectbox(
                    "Theme:",
                    options=theme_options,
                    index=current_theme_idx,
                    help="Choose light or dark theme (requires page refresh)"
                )
                if new_theme != config.theme_base:
                    config_manager.update_config({"theme_base": new_theme})
                    st.info("🔄 Refresh the page to apply the new theme")
                
                # Sidebar state
                sidebar_options = ["expanded", "collapsed", "auto"]
                try:
                    current_sidebar_idx = sidebar_options.index(config.sidebar_state)
                except ValueError:
                    current_sidebar_idx = 0
                
                new_sidebar_state = st.selectbox(
                    "Sidebar State:",
                    options=sidebar_options,
                    index=current_sidebar_idx,
                    help="Default state of the sidebar on page load"
                )
                if new_sidebar_state != config.sidebar_state:
                    config_manager.update_config({"sidebar_state": new_sidebar_state})
            
            # API Key Management Section
            st.header("🔑 API Key Management")
            api_key_valid = (config.google_api_key and 
                           config.google_api_key != "<enter google api key>" and
                           config.google_api_key.startswith("AIzaSy") and 
                           len(config.google_api_key) > 30)
            
            # Status display
            status_color = "#4caf50" if api_key_valid else "#f44336"
            status_text = "Valid & Active" if api_key_valid else "Missing/Invalid"
            status_icon = "✅" if api_key_valid else "❌"
            
            st.markdown(f"""
            <div style="padding: 12px; border: 2px solid {status_color}; border-radius: 8px; background-color: {'#e8f5e8' if api_key_valid else '#ffebee'};">
                <div style="font-size: 14px; font-weight: 500; color: {status_color};">
                    {status_icon} API Key Status: {status_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # API Key management
            key_section_expanded = not api_key_valid or not is_config_complete
            with st.expander("🔧 Manage API Key", expanded=key_section_expanded):
                current_key = config.google_api_key if config.google_api_key else ""
                
                if current_key and len(current_key) > 12:
                    masked_key = f"{current_key[:8]}...{current_key[-4:]}"
                    st.code(f"Current: {masked_key}", language=None)
                
                # API key input form
                with st.form("api_key_form"):
                    new_api_key = st.text_input(
                        "Enter Google API Key:",
                        placeholder="AIzaSy...",
                        type="password",
                        help="Get your API key from Google AI Studio: https://aistudio.google.com/app/apikey"
                    )
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        save_key = st.form_submit_button("💾 Save Key", use_container_width=True)
                    with col2:
                        clear_key = st.form_submit_button("🗑️ Clear Key", use_container_width=True)
                    
                    if save_key and new_api_key:
                        if new_api_key.startswith("AIzaSy") and len(new_api_key) > 30:
                            config_manager.update_config({"google_api_key": new_api_key})
                            st.success("✅ API Key saved successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Invalid API key format. Should start with 'AIzaSy' and be longer than 30 characters.")
                    
                    if clear_key and current_key:
                        config_manager.update_config({"google_api_key": ""})
                        st.warning("⚠️ API Key cleared.")
                        st.rerun()
            
            # Chat Controls
            st.header("🎮 Chat Controls")
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.messages = []
                    if 'chat_app' in st.session_state:
                        chat_app.clear_memory()
                    st.success("Chat history cleared!")
                    st.rerun()
            
            with col2:
                if st.button("🔄 Reload Config", use_container_width=True):
                    try:
                        config_manager.load_config()
                        st.success("Configuration reloaded!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to reload config: {str(e)}")
            
            # Configuration Details
            with st.expander("📋 Configuration Details", expanded=False):
                tab1, tab2 = st.tabs(["Model", "UI"])
                
                with tab1:
                    st.write(f"**Model:** {config.gemini_model}")
                    st.write(f"**Chat History:** {'Enabled' if config.include_chat_history else 'Disabled'}")
                    if config.include_chat_history:
                        st.write(f"**Max History:** {config.max_history_messages} messages")
                    st.write(f"**URL Context:** {'Enabled' if config.url_context_tool else 'Disabled'}")
                    st.write(f"**Policy URLs:** {len(config.policy_urls)} configured")
                
                with tab2:
                    st.write(f"**Title:** {config.page_title}")
                    st.write(f"**Icon:** {config.page_icon}")
                    st.write(f"**Layout:** {config.layout}")
                    st.write(f"**Theme:** {config.theme_base}")
                    st.write(f"**Sidebar:** {config.sidebar_state}")
        
        # Main chat interface
        if is_config_complete:
            # Show chat app status
            status = chat_app.get_status()
            if status["ready"]:
                # Memory stats (if chat history exists)
                if len(st.session_state.messages) > 0:
                    memory_stats = chat_app.get_memory_stats()
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Messages", memory_stats["total_messages"])
                    col2.metric("Conversations", memory_stats["user_messages"])
                    col3.metric("Avg Length", f"{memory_stats['average_message_length']} chars")
                
                # Display chat history
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                
                # Chat input
                if prompt := st.chat_input("Ask about medical policies...", key="main_chat_input"):
                    # Validate input
                    if len(prompt.strip()) == 0:
                        st.warning("Please enter a valid question.")
                        st.stop()
                    
                    # Add user message
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # Generate response
                    with st.chat_message("assistant"):
                        with st.spinner("🔍 Analyzing..."):
                            try:
                                response = chat_app.query_with_context(prompt)
                                st.markdown(response)
                                
                                # Add to session and memory
                                chat_app.add_to_memory(prompt, response)
                                st.session_state.messages.append({"role": "assistant", "content": response})
                                
                            except Exception as e:
                                error_msg = f"❌ Error generating response: {str(e)}"
                                st.error(error_msg)
                                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            else:
                st.error(f"❌ Chat app not ready: {status.get('last_error', 'Unknown error')}")
                st.info("Please check your configuration in the sidebar.")
        
        else:
            # Setup guidance for incomplete configuration
            st.markdown("---")
            st.markdown("### 🔧 Complete Setup to Start Chatting")
            
            setup_steps = []
            if not status.get("api_key_valid", False):
                setup_steps.append("🔑 **Set up your Google API Key** in the sidebar")
            if len(config.policy_urls) == 0:
                setup_steps.append("📋 **Add at least one policy URL** to analyze")
            
            for step in setup_steps:
                st.markdown(f"- {step}")
            
            st.markdown("Once setup is complete, you'll be able to ask questions about medical policies!")
            
            # Example questions
            with st.expander("💡 Example Questions", expanded=True):
                st.markdown("""
                - "What are the coverage criteria for this procedure?"
                - "Are there any age restrictions mentioned in the policy?"
                - "What documentation is required for approval?"
                - "What are the exclusions for this treatment?"
                - "Summarize the key points of this policy"
                """)
            
            # Disabled chat input
            st.chat_input("Complete setup to enable chat...", disabled=True, key="disabled_chat_input")
    
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.info("Please refresh the page or contact support if the issue persists.")
        if st.button("🔄 Refresh Application"):
            st.rerun()

if __name__ == "__main__":
    main()