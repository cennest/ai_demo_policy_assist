"""
File metadata validation system with API key hashing and automatic expiration
"""

import hashlib
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, TYPE_CHECKING
import json
import logging
from pathlib import Path

if TYPE_CHECKING:
    from google_gen_ai_client import GoogleGenAI

class FileMetadataValidator:
    """
    Validates file metadata using API key hashes and manages automatic expiration.
    
    Features:
    - Hash API keys to validate file metadata integrity
    - Automatically expire file metadata after 24 hours
    - Secure validation without storing sensitive API keys
    """
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
    def generate_api_key_hash(self, api_key: str, salt: str = None) -> str:
        """
        Generate a secure hash of the API key for validation purposes.
        
        Args:
            api_key: The API key to hash
            salt: Optional salt for additional security
            
        Returns:
            Hexadecimal hash string
        """
        if not api_key:
            raise ValueError("API key cannot be empty")
            
        # Use a default salt if none provided
        if salt is None:
            salt = "file_metadata_validation_salt_2025"
            
        # Create hash using SHA-256
        hash_input = f"{api_key}{salt}".encode('utf-8')
        return hashlib.sha256(hash_input).hexdigest()
    
    def is_expired(self, file_object: Dict[str, Any]) -> bool:
        """
        Check if a file object has expired (older than 24 hours).
        
        Args:
            file_object: File object metadata dictionary
            
        Returns:
            True if expired, False otherwise
        """
        if not file_object or 'expiration_time' not in file_object:
            return True
            
        try:
            # Parse expiration time
            expiration_str = file_object['expiration_time']
            if expiration_str.endswith('Z'):
                expiration_time = datetime.fromisoformat(expiration_str[:-1]).replace(tzinfo=timezone.utc)
            else:
                expiration_time = datetime.fromisoformat(expiration_str)
                if expiration_time.tzinfo is None:
                    expiration_time = expiration_time.replace(tzinfo=timezone.utc)
            
            # Check if expired
            current_time = datetime.now(timezone.utc)
            return current_time >= expiration_time
            
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Error parsing expiration time: {e}")
            return True
    
    def validate_file_metadata(self, url: str, metadata: Dict[str, Any], current_api_key: str) -> bool:
        """
        Validate file metadata using API key hash and expiration check.
        
        Args:
            url: The URL associated with the file
            metadata: File metadata dictionary
            current_api_key: Current API key for validation
            
        Returns:
            True if metadata is valid and not expired, False otherwise
        """
        if not metadata or 'file_object' not in metadata:
            return False
            
        file_object = metadata['file_object']
        
        # Check if file has expired
        if self.is_expired(file_object):
            self.logger.info(f"File metadata expired for URL: {url}")
            return False
            
        # Check API key hash validation
        stored_hash = metadata.get('api_key_hash')
        if stored_hash:
            current_hash = self.generate_api_key_hash(current_api_key)
            if stored_hash != current_hash:
                self.logger.warning(f"API key hash mismatch for URL: {url}")
                return False
        else:
            # No stored hash means file was uploaded without proper validation - reset it
            self.logger.info(f"No API key hash found for URL: {url} - invalidating for security")
            return False
                
        return True
    
    def add_metadata_validation(self, metadata: Dict[str, Any], api_key: str) -> Dict[str, Any]:
        """
        Add validation metadata to file metadata dictionary.
        
        Args:
            metadata: Existing metadata dictionary
            api_key: API key to hash for validation
            
        Returns:
            Updated metadata dictionary with validation info
        """
        updated_metadata = metadata.copy()
        
        # Add API key hash for validation
        updated_metadata['api_key_hash'] = self.generate_api_key_hash(api_key)
        
        # Add validation timestamp
        updated_metadata['validation_timestamp'] = datetime.now(timezone.utc).isoformat()
        
        return updated_metadata
    
    def cleanup_expired_files(self, force_cleanup: bool = False) -> List[str]:
        """
        Clean up expired file metadata from configuration.
        
        Args:
            force_cleanup: Force cleanup regardless of last cleanup time
            
        Returns:
            List of URLs that were cleaned up
        """
        if not self.config_manager:
            self.logger.warning("No config manager available for cleanup")
            return []
            
        config = self.config_manager.get_config()
        cleaned_urls = []
        
        # Check each policy URL for expired files
        urls_to_clean = []
        for url, metadata in config.policy_urls.items():
            if 'file_object' in metadata:
                if self.is_expired(metadata['file_object']):
                    urls_to_clean.append(url)
        
        # Clean up expired files
        for url in urls_to_clean:
            metadata = config.policy_urls[url].copy()
            
            # Remove file-related metadata
            if 'file_object' in metadata:
                del metadata['file_object']
            if 'file_uri' in metadata:
                del metadata['file_uri']
            if 'api_key_hash' in metadata:
                del metadata['api_key_hash']
            if 'validation_timestamp' in metadata:
                del metadata['validation_timestamp']
                
            # Update configuration
            self.config_manager.save_policy_url(url, metadata)
            cleaned_urls.append(url)
            self.logger.info(f"Cleaned up expired file metadata for: {url}")
        
        return cleaned_urls
    
    def get_expiration_info(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get expiration information for file metadata.
        
        Args:
            metadata: File metadata dictionary
            
        Returns:
            Dictionary with expiration information
        """
        if not metadata or 'file_object' not in metadata:
            return {'status': 'no_file', 'expired': True}
            
        file_object = metadata['file_object']
        
        try:
            expiration_str = file_object.get('expiration_time')
            if not expiration_str:
                return {'status': 'no_expiration', 'expired': True}
                
            # Parse expiration time
            if expiration_str.endswith('Z'):
                expiration_time = datetime.fromisoformat(expiration_str[:-1]).replace(tzinfo=timezone.utc)
            else:
                expiration_time = datetime.fromisoformat(expiration_str)
                if expiration_time.tzinfo is None:
                    expiration_time = expiration_time.replace(tzinfo=timezone.utc)
            
            current_time = datetime.now(timezone.utc)
            is_expired = current_time >= expiration_time
            time_remaining = expiration_time - current_time
            
            return {
                'status': 'valid',
                'expired': is_expired,
                'expiration_time': expiration_time.isoformat(),
                'current_time': current_time.isoformat(),
                'time_remaining_seconds': time_remaining.total_seconds(),
                'time_remaining_hours': time_remaining.total_seconds() / 3600
            }
            
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Error parsing expiration info: {e}")
            return {'status': 'parse_error', 'expired': True, 'error': str(e)}
    
    def schedule_daily_cleanup(self):
        """
        Schedule automatic daily cleanup of expired files.
        This should be called at application startup.
        """
        # For now, just perform immediate cleanup
        # In a production environment, you might want to use a task scheduler
        cleaned_urls = self.cleanup_expired_files()
        if cleaned_urls:
            self.logger.info(f"Daily cleanup completed. Cleaned {len(cleaned_urls)} expired files.")
        
        return cleaned_urls

    def delete_all_gemini_files_and_reset_metadata(self, google_client: "GoogleGenAI") -> Dict[str, Any]:
        """
        Delete all files from Gemini Files API and reset all metadata in config.
        
        Args:
            google_client: GoogleGenAI client instance for API calls
            
        Returns:
            Dictionary with deletion results
        """
        if not self.config_manager:
            return {"status": "error", "message": "No config manager available"}
        
        results = {
            "files_deleted": 0,
            "files_failed": 0,
            "urls_reset": 0,
            "deleted_files": [],
            "failed_files": [],
            "reset_urls": []
        }
        
        config = self.config_manager.get_config()
        
        # First, list all files currently in Gemini API (not from cache)
        try:
            current_files = self.list_all_gemini_files(google_client)
            self.logger.info(f"Found {len(current_files)} files in Gemini API to delete")
        except Exception as e:
            self.logger.error(f"Failed to list files from Gemini API: {e}")
            current_files = []
        
        # Delete all files found in Gemini API
        for file_info in current_files:
            file_name = file_info.get("name", "")
            if not file_name:
                continue
                
            try:
                # Try to delete the file from Gemini
                success = google_client.delete_file_sync(file_name)
                if success:
                    results["deleted_files"].append(file_name)
                    results["files_deleted"] += 1
                    self.logger.info(f"Deleted Gemini file: {file_name}")
                else:
                    results["failed_files"].append(file_name)
                    results["files_failed"] += 1
                    self.logger.warning(f"Failed to delete Gemini file: {file_name}")
                    
            except Exception as e:
                self.logger.warning(f"Failed to delete Gemini file {file_name}: {e}")
                results["failed_files"].append(file_name)
                results["files_failed"] += 1
        
        # Reset all metadata regardless of deletion success
        for url, metadata in config.policy_urls.items():
            original_metadata = metadata.copy()
            
            # Remove all file-related metadata
            fields_to_remove = ['file_object', 'file_uri', 'api_key_hash', 'validation_timestamp']
            reset_needed = False
            
            for field in fields_to_remove:
                if field in metadata:
                    del metadata[field]
                    reset_needed = True
            
            if reset_needed:
                self.config_manager.save_policy_url(url, metadata)
                results["reset_urls"].append(url)
                results["urls_reset"] += 1
                self.logger.info(f"Reset metadata for URL: {url}")
        
        return results
    
    def list_all_gemini_files(self, google_client: "GoogleGenAI") -> List[Dict[str, Any]]:
        """
        List all files currently stored in Gemini Files API.
        
        Args:
            google_client: GoogleGenAI client instance
            
        Returns:
            List of file information dictionaries
        """
        files = []
        
        try:
            # Use direct client access with proper pagination handling
            files_response = google_client.client.files.list()
            # files_response is a Pager object, convert to list
            file_objects = list(files_response)
            
            for file_obj in file_objects:
                # Extract file information
                file_dict = {
                    'name': getattr(file_obj, 'name', ''),
                    'display_name': getattr(file_obj, 'display_name', ''),
                    'uri': getattr(file_obj, 'uri', ''),
                    'mime_type': getattr(file_obj, 'mime_type', ''),
                    'size_bytes': getattr(file_obj, 'size_bytes', 0),
                    'create_time': str(getattr(file_obj, 'create_time', '')),
                    'update_time': str(getattr(file_obj, 'update_time', '')),
                    'expiration_time': str(getattr(file_obj, 'expiration_time', '')),
                    'state': getattr(file_obj, 'state', ''),
                    'sha256_hash': getattr(file_obj, 'sha256_hash', ''),
                }
                files.append(file_dict)
        except Exception as e:
            self.logger.error(f"Failed to list Gemini files: {e}")
            
        return files


class SecureFileManager:
    """
    Enhanced file manager with security features for the policy assistant.
    """
    
    def __init__(self, config_manager, api_key: str):
        self.config_manager = config_manager
        self.api_key = api_key
        self.validator = FileMetadataValidator(config_manager)
        self.logger = logging.getLogger(__name__)
    
    def secure_save_file_metadata(self, url: str, metadata: Dict[str, Any]) -> bool:
        """
        Securely save file metadata with validation hash.
        
        Args:
            url: URL associated with the file
            metadata: File metadata to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Add validation metadata
            secure_metadata = self.validator.add_metadata_validation(metadata, self.api_key)
            
            # Save to configuration
            return self.config_manager.save_policy_url(url, secure_metadata)
            
        except Exception as e:
            self.logger.error(f"Error saving secure file metadata: {e}")
            return False
    
    def get_valid_file_metadata(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Get file metadata only if it's valid and not expired.
        
        Args:
            url: URL to get metadata for
            
        Returns:
            Valid metadata dictionary or None if invalid/expired
        """
        config = self.config_manager.get_config()
        metadata = config.policy_urls.get(url)
        
        if not metadata:
            return None
            
        # Validate metadata
        if self.validator.validate_file_metadata(url, metadata, self.api_key):
            return metadata
        else:
            # Clean up invalid metadata
            self.cleanup_invalid_metadata(url)
            return None
    
    def cleanup_invalid_metadata(self, url: str):
        """
        Clean up invalid or expired metadata for a specific URL.
        
        Args:
            url: URL to clean up
        """
        config = self.config_manager.get_config()
        metadata = config.policy_urls.get(url)
        
        if metadata:
            cleaned_metadata = metadata.copy()
            
            # Remove file-related fields
            for field in ['file_object', 'file_uri', 'api_key_hash', 'validation_timestamp']:
                if field in cleaned_metadata:
                    del cleaned_metadata[field]
            
            self.config_manager.save_policy_url(url, cleaned_metadata)
            self.logger.info(f"Cleaned up invalid metadata for: {url}")
    
    def perform_security_check(self) -> Dict[str, Any]:
        """
        Perform a comprehensive security check on all file metadata.
        
        Returns:
            Dictionary with security check results
        """
        config = self.config_manager.get_config()
        results = {
            'total_urls': len(config.policy_urls),
            'valid_files': 0,
            'expired_files': 0,
            'invalid_hash_files': 0,
            'missing_files': 0,
            'cleaned_urls': []
        }
        
        for url, metadata in config.policy_urls.items():
            if 'file_object' not in metadata:
                results['missing_files'] += 1
                continue
                
            # Check expiration
            if self.validator.is_expired(metadata['file_object']):
                results['expired_files'] += 1
                self.cleanup_invalid_metadata(url)
                results['cleaned_urls'].append(url)
                continue
                
            # Check API key hash
            stored_hash = metadata.get('api_key_hash')
            if stored_hash:
                current_hash = self.validator.generate_api_key_hash(self.api_key)
                if stored_hash != current_hash:
                    results['invalid_hash_files'] += 1
                    self.cleanup_invalid_metadata(url)
                    results['cleaned_urls'].append(url)
                    continue
            else:
                # No hash means insecure file - clean it up
                results['invalid_hash_files'] += 1
                self.cleanup_invalid_metadata(url)
                results['cleaned_urls'].append(url)
                continue
                    
            results['valid_files'] += 1
        
        self.logger.info(f"Security check completed: {results}")
        return results
    
    def delete_all_files_and_reset(self, google_client: "GoogleGenAI") -> Dict[str, Any]:
        """
        Delete all Gemini files and reset metadata using the validator.
        
        Args:
            google_client: GoogleGenAI client instance
            
        Returns:
            Dictionary with operation results
        """
        return self.validator.delete_all_gemini_files_and_reset_metadata(google_client)
    
    def list_gemini_files(self, google_client: "GoogleGenAI") -> List[Dict[str, Any]]:
        """
        List all files in Gemini Files API.
        
        Args:
            google_client: GoogleGenAI client instance
            
        Returns:
            List of file information
        """
        return self.validator.list_all_gemini_files(google_client)