import streamlit as st
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import BaseMessage
from config import ConfigManager

class PolicyChatApp:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        self.setup_gemini()
        self.chat_history: List[BaseMessage] = []
    
    def setup_gemini(self):
        """Initialize ChatGoogleGenerativeAI with LangChain"""
        if not self.config.google_api_key:
            st.warning("Please set GOOGLE_API_KEY before starting chat")
            return
        
        # Initialize LangChain ChatGoogleGenerativeAI
        self.model = ChatGoogleGenerativeAI(
            model=self.config.gemini_model,
            google_api_key=self.config.google_api_key,
            temperature=0.1,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
    
    def query_with_context(self, user_message: str) -> str:
        """Query Gemini with URL context and LangChain message history for medical policy information"""
        try:
            # Build messages list using LangChain message objects
            messages = []
            
            # Add system message with medical policy context
            # Use configurable system prompt if provided, otherwise use default
            if self.config.system_prompt.strip():
                system_prompt = self.config.system_prompt
            else:
                system_prompt = """Your job is to extract the most relevant information from the provided context URLs to answer user questions. Use only what is explicitly stated in those documents — do not make assumptions, guesses, or provide answers beyond the given evidence. If the answer is not clearly supported by the context, respond with "Not mentioned in the provided policy." Always refer to the chat history to maintain context."""
            messages.append(SystemMessage(content=system_prompt))
            
            # Add policy URLs context if enabled
            policy_urls = self.config.policy_urls
            if self.config.url_context_tool and policy_urls:
                url_context = "Policy URLs to analyze:\n" + "\n".join(f"- {url}" for url in policy_urls)
                messages.append(HumanMessage(content=url_context))

            # Add recent conversation history
            if self.chat_history:
                max_messages = self.config.max_history_messages
                recent_messages = self.chat_history[-max_messages:]
                messages.extend(recent_messages)
            
            
                 
            messages.append(HumanMessage(content=user_message))
            
            # Invoke the LangChain model with messages
            response = self.model.invoke(messages)
            
            return response.content
            
        except Exception as e:
            return f"Error querying Gemini: {str(e)}"
    
    def add_to_memory(self, user_message: str, ai_response: str):
        """Add messages to conversation history"""
        self.chat_history.append(HumanMessage(content=user_message))
        self.chat_history.append(AIMessage(content=ai_response))
        
    def clear_memory(self):
        """Clear conversation history"""
        self.chat_history.clear()

def main():
    # Initialize config manager
    if "config_manager" not in st.session_state:
        st.session_state.config_manager = ConfigManager()
    
    config_manager = st.session_state.config_manager
    config = config_manager.get_config()
    
    # Configure page with settings from config
    st.set_page_config(
        page_title=config.page_title,
        page_icon=config.page_icon,
        layout=config.layout,
        initial_sidebar_state=config.sidebar_state
    )
    
    # Custom CSS for better theme alignment and emoji sizing
    st.markdown(f"""
    <style>
    /* Reduce emoji size in buttons */
    .stButton button {{
        font-size: 14px !important;
        padding: 0.25rem 0.5rem !important;
        height: 2.5rem !important;
    }}
    
    /* Specific styling for delete buttons (× symbol) */
    .stButton button[title*="Delete"] {{
        font-size: 16px !important;
        padding: 0.2rem !important;
        height: 2rem !important;
        background-color: #ff4b4b !important;
        color: white !important;
        border: none !important;
    }}
    
    .stButton button[title*="Delete"]:hover {{
        background-color: #ff6b6b !important;
    }}
    
    /* Customize sidebar width and padding */
    .css-1d391kg {{
        padding-top: 1rem;
        width: 350px !important;
    }}
    
    /* Better alignment for status messages */
    .stAlert {{
        font-size: 14px;
        padding: 0.5rem 1rem;
    }}
    
    /* Consistent emoji sizing in headers - smaller emojis */
    h1 {{
        font-size: 1.8rem !important;
        line-height: 1.2;
    }}
    
    h2 {{
        font-size: 1.4rem !important;
        line-height: 1.2;
    }}
    
    h3 {{
        font-size: 1.2rem !important;
        line-height: 1.2;
    }}
    
    /* Better spacing for configuration sections */
    .stExpander {{
        margin-bottom: 0.5rem;
    }}
    
    /* Control emoji size in metrics */
    .metric-container {{
        font-size: 14px;
    }}
    
    /* Explicitly control metric value size (API Key Status icons) */
    [data-testid="metric-container"] {{
        font-size: 14px !important;
    }}
    
    [data-testid="metric-container"] [data-testid="stMetricValue"] {{
        font-size: 16px !important;
        line-height: 1.2 !important;
    }}
    
    [data-testid="metric-container"] [data-testid="stMetricLabel"] {{
        font-size: 14px !important;
    }}
    
    /* Compact input fields */
    .stTextInput input {{
        padding: 0.5rem !important;
        font-size: 14px !important;
    }}
    
    /* Better chat input styling */
    .stChatInput input {{
        font-size: 16px !important;
    }}
    
    /* Theme-based styling */
    .stApp {{
        background-color: {'#fafafa' if config.theme_base == 'light' else '#0e1117'};
    }}
    </style>
    """, unsafe_allow_html=True)
    
    st.title(f"{config.page_icon} {config.page_title}")
    st.markdown("*Ask questions about medical necessity criteria and policy requirements*")
    
    # Check if configuration is complete
    config_errors = config_manager.validate_config()
    is_config_complete = len(config_errors) == 0
    
    # Show configuration status
    if not is_config_complete:
        st.error("⚠️ **Configuration Required** - Please complete the settings in the sidebar before starting chat")
        for field, error in config_errors.items():
            st.error(f"❌ {error}")
        st.info("👈 Use the sidebar to configure your settings")
        
        # Force sidebar to be expanded on mobile
        if not st.session_state.get("sidebar_expanded", False):
            st.session_state.sidebar_expanded = True
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_app" not in st.session_state:
        st.session_state.chat_app = PolicyChatApp(config_manager)
    
    # Sidebar for policy URL configuration
    with st.sidebar:
        # Show urgent configuration status at top
        if not is_config_complete:
            st.error("🔧 **Setup Required**")
            st.markdown("**Complete the following to start chatting:**")
            for field, error in config_errors.items():
                st.markdown(f"• {error}")
            st.divider()
        else:
            st.success("✅ **Configuration Complete**")
            st.divider()
        
        st.header("📋 Policy Configuration")
        
        # Add new policy URL
        new_url = st.text_input("Add Policy URL:", placeholder="https://example.com/policy")
        if st.button("Add URL") and new_url:
            if config_manager.add_policy_url(new_url):
                st.success(f"Added: {new_url}")
                st.rerun()
            else:
                st.warning("URL already exists or is empty")
        
        # Display current URLs
        st.subheader("Current Policy URLs:")
        for i, url in enumerate(config.policy_urls):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.text(url[:50] + "..." if len(url) > 50 else url)
            with col2:
                # Use a smaller × symbol instead of emoji
                if st.button("×", key=f"remove_{i}", use_container_width=True, help="Delete URL"):
                    config_manager.remove_policy_url(url)
                    st.rerun()
        
        st.divider()
        
        # Configuration settings
        st.header("⚙️ Settings")
        
        # System prompt setting
        st.subheader("💬 System Prompt")
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
        
        # Max history messages setting
        new_max_messages = st.number_input(
            "Max History Messages:", 
            min_value=1, 
            max_value=50, 
            value=config.max_history_messages
        )
        if new_max_messages != config.max_history_messages:
            config_manager.update_config({"max_history_messages": new_max_messages})
        
        # Gemini model setting
        model_options = ["gemini-2.5-flash", "gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"]
        current_model_idx = model_options.index(config.gemini_model) if config.gemini_model in model_options else 0
        new_model = st.selectbox(
            "Gemini Model:", 
            options=model_options,
            index=current_model_idx
        )
        if new_model != config.gemini_model:
            config_manager.update_config({"gemini_model": new_model})
            # Reinitialize chat app with new model
            st.session_state.chat_app = PolicyChatApp(config_manager)
        
        # URL Context Tool setting
        new_url_context = st.checkbox(
            "Enable URL Context Tool",
            value=config.url_context_tool,
            help="Use Gemini's URL context feature to analyze policy URLs directly"
        )
        if new_url_context != config.url_context_tool:
            config_manager.update_config({"url_context_tool": new_url_context})
        
        # Theme settings
        theme_options = ["light", "dark"]
        current_theme_idx = theme_options.index(config.theme_base) if config.theme_base in theme_options else 0
        new_theme = st.selectbox(
            "Theme:",
            options=theme_options,
            index=current_theme_idx,
            help="Choose light or dark theme (requires page refresh)"
        )
        if new_theme != config.theme_base:
            config_manager.update_config({"theme_base": new_theme})
            st.info("🔄 Refresh the page to apply the new theme")
        
        # Sidebar state setting
        sidebar_options = ["expanded", "collapsed", "auto"]
        current_sidebar_idx = sidebar_options.index(config.sidebar_state) if config.sidebar_state in sidebar_options else 0
        new_sidebar_state = st.selectbox(
            "Sidebar State:",
            options=sidebar_options,
            index=current_sidebar_idx,
            help="Default state of the sidebar on page load"
        )
        if new_sidebar_state != config.sidebar_state:
            config_manager.update_config({"sidebar_state": new_sidebar_state})
        
        # Chat controls
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_app.clear_memory()
            st.rerun()
        
        # Configuration info
        st.header("ℹ️ Current Configuration")
        
        # API Key Management
        st.subheader("🔑 Google API Key")
        api_key_valid = config.google_api_key and config.google_api_key != "<enter google api key>" and config.google_api_key.startswith("AIzaSy") and len(config.google_api_key) > 30
        api_status_icon = "✅" if api_key_valid else "❌"
        api_status_text = "Set" if api_key_valid else "Missing/Invalid"
        
        # Custom status display with controlled icon size
        st.markdown(f"""
        <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f8f9fa;">
            <div style="font-size: 12px; color: #666; margin-bottom: 4px;">API Key Status</div>
            <div style="font-size: 14px;">
                <span style="font-size: 14px;">{api_status_icon}</span> 
                <span style="font-weight: 500;">{api_status_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # API Key input section - expand if key is missing/invalid or settings are incomplete
        key_section_expanded = not api_key_valid or not is_config_complete
        with st.expander("🔧 Manage API Key", expanded=key_section_expanded):
            current_key = config.google_api_key if config.google_api_key else ""
            masked_key = f"{current_key[:8]}...{current_key[-4:]}" if len(current_key) > 12 else current_key
            
            if current_key:
                st.write(f"**Current Key:** `{masked_key}`")
            
            # Input for new API key
            new_api_key = st.text_input(
                "Enter Google API Key:",
                placeholder="AIzaSy...",
                type="password",
                help="Get your API key from Google AI Studio: https://aistudio.google.com/app/apikey"
            )
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("💾 Save API Key", use_container_width=True) and new_api_key:
                    if new_api_key.startswith("AIzaSy") and len(new_api_key) > 30:
                        config_manager.update_config({"google_api_key": new_api_key})
                        # Reinitialize chat app with new API key
                        st.session_state.chat_app = PolicyChatApp(config_manager)
                        st.success("✅ API Key saved successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid API key format. Should start with 'AIzaSy' and be longer than 30 characters.")
            
            with col2:
                if st.button("🗑️ Clear API Key", use_container_width=True) and current_key:
                    config_manager.update_config({"google_api_key": ""})
                    st.warning("⚠️ API Key cleared. Please set a new key to use the application.")
                    st.rerun()
        
        # Configuration details in expandable section
        with st.expander("📋 Configuration Details", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Settings")
                st.write(f"**Model:** {config.gemini_model}")
                st.write(f"**Max History:** {config.max_history_messages} messages")
                st.write(f"**URL Context:** {'Enabled' if config.url_context_tool else 'Disabled'}")
            
            with col2:
                st.subheader("UI Settings")
                st.write(f"**Title:** {config.page_title}")
                st.write(f"**Icon:** {config.page_icon}")
                st.write(f"**Layout:** {config.layout}")
        
        # Policy URLs section
        st.subheader("📎 Policy URLs")
        if config.policy_urls:
            for i, url in enumerate(config.policy_urls, 1):
                with st.expander(f"URL {i}: {url[:30]}...", expanded=False):
                    st.code(url, language=None)
        else:
            st.info("No policy URLs configured yet")
        
        # Config file info
        st.subheader("⚙️ Config File")
        st.code("config.json", language=None)
        
        if st.button("🔄 Reload Configuration", use_container_width=True):
            config_manager.load_config()
            st.success("Configuration reloaded!")
            st.rerun()
    
    # Main chat interface
    if is_config_complete:
        # Display chat history only if config is complete
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input - only show if config is complete
        if prompt := st.chat_input("Ask about medical policies..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response using Gemini with policy context
            with st.chat_message("assistant"):
                with st.spinner("Analyzing policies and generating response..."):
                    response = st.session_state.chat_app.query_with_context(prompt)
                    st.markdown(response)
            
            # Add to memory and chat history
            st.session_state.chat_app.add_to_memory(prompt, response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        # Show placeholder content when settings are incomplete
        st.markdown("---")
        st.markdown("### 🔧 Complete Setup to Start Chatting")
        st.markdown("""
        To begin using the Policy Chat Assistant:
        
        1. **Set up your Google API Key** in the sidebar
        2. **Add at least one policy URL** to analyze
        3. **Review your settings** and ensure everything is configured
        
        Once setup is complete, you'll be able to ask questions about medical policies and get AI-powered responses.
        """)
        
        # Show example of what they can do once configured
        with st.expander("💡 What you can do once configured", expanded=True):
            st.markdown("""
            **Example questions you can ask:**
            - "What are the coverage criteria for this procedure?"
            - "Are there any age restrictions mentioned in the policy?"
            - "What documentation is required for approval?"
            - "What are the exclusions for this treatment?"
            """)
        
        # Disabled chat input to show it's blocked
        st.text_input("Chat input (disabled - complete setup first)", disabled=True, placeholder="Complete setup to enable chat...")

if __name__ == "__main__":
    main()