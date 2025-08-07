"""Configuration management for Policy Chat Assistant"""

import os
import json
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class AppConfig:
    """Application configuration"""
    # API Settings
    google_api_key: str = "<enter google api key>"
    gemini_model: str = "gemini-2.5-flash"
    url_context_tool: bool = True
    
    # Chat Settings  
    max_history_messages: int = 50
    system_prompt: str = ""
    include_chat_history: bool = True
    
    # Policy URLs
    policy_urls: List[str] = None
    
    # UI Settings
    page_title: str = "Policy Chat Assistant"
    page_icon: str = "📋"
    layout: str = "wide"
    sidebar_state: str = "expanded"
    theme_base: str = "light"  # "light" or "dark"
    
    def __post_init__(self):
        if self.policy_urls is None:
            self.policy_urls = []


class ConfigManager:
    """Manages configuration loading and saving"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self.config = AppConfig()
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file and environment variables"""
        # Load from file if exists
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    # Update config with file data
                    for key, value in config_data.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
            except (json.JSONDecodeError, IOError, UnicodeDecodeError) as e:
                print(f"Warning: Could not load config file: {e}")
        
        # Override with environment variables
        self.config.google_api_key = os.getenv("GOOGLE_API_KEY", self.config.google_api_key)
        
        # If still no API key, try from config file
        if not self.config.google_api_key:
            print("Warning: GOOGLE_API_KEY not found in environment or config")
    
    def save_config(self, include_api_key: bool = True) -> None:
        """Save current configuration to file"""
        try:
            config_dict = asdict(self.config)
            # Only save API key to file if explicitly requested
            if not include_api_key:
                config_dict.pop('google_api_key', None)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Warning: Could not save config file: {e}")
    
    def add_policy_url(self, url: str) -> bool:
        """Add a policy URL to the configuration"""
        if url and url not in self.config.policy_urls:
            self.config.policy_urls.append(url)
            self.save_config()
            return True
        return False
    
    def remove_policy_url(self, url: str) -> bool:
        """Remove a policy URL from the configuration"""
        if url in self.config.policy_urls:
            self.config.policy_urls.remove(url)
            self.save_config()
            return True
        return False
    
    def get_config(self) -> AppConfig:
        """Get current configuration"""
        return self.config
    
    def validate_config(self) -> Dict[str, str]:
        """Validate configuration and return validation errors"""
        errors = {}
        
        # Check API key
        if not self.config.google_api_key or self.config.google_api_key == "<enter google api key>":
            errors["google_api_key"] = "Google API key is required"
        elif not (self.config.google_api_key.startswith("AIzaSy") and len(self.config.google_api_key) > 30):
            errors["google_api_key"] = "Invalid Google API key format"
        
        # Check policy URLs
        if not self.config.policy_urls:
            errors["policy_urls"] = "At least one policy URL is required"
        
        return errors
    
    def is_config_complete(self) -> bool:
        """Check if configuration is complete and valid"""
        return len(self.validate_config()) == 0
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Save API key to file if it's being updated
        include_api_key = 'google_api_key' in updates
        self.save_config(include_api_key=include_api_key)