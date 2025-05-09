import re
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass
import json
import os
import yaml
import requests
from time import time

@dataclass
class ModerationResult:
    severity: float
    categories: Dict[str, bool]
    action: str
    message: str
    timeout_duration: Optional[int] = None

class ContentModerator:
    def __init__(self, config: dict):
        self.config_path = 'config.yaml'  # Store config path for reloading
        self.last_api_call = 0  # Rate limiting
        self.api_call_interval = 1  # Minimum seconds between API calls
        self.update_config(config)
        print("Content moderator initialized")
    
    def update_config(self, config: dict):
        """Update moderator with new config"""
        self.config = config
        self.moderation_config = config.get('moderation', {})
        self.settings = self.moderation_config.get('settings', {})
        
        # Initialize violation tracking
        self.user_violations = {}
        
        # Get API settings
        self.api_key = os.getenv('HUGGINGFACE_API_KEY')  # Get API key from environment variable
        self.use_api = bool(self.api_key) and self.moderation_config.get('use_api', True)
        self.api_url = "https://api-inference.huggingface.co/models/facebook/roberta-hate-speech-dynabench-r4-target"
        
        self.load_word_lists()
        print("Moderator config updated")
    
    def reload_config(self):
        """Reload config from file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                self.update_config(config)
                return True
        except Exception as e:
            print(f"Error reloading config: {e}")
            return False
    
    def load_word_lists(self):
        """Load word lists from files or create default ones"""
        self.word_lists = {
            'profanity': set([
                'fuck', 'shit', 'ass', 'bitch', 'dick', 'pussy', 'cock', 'cunt',
                'bastard', 'motherfucker', 'whore', 'slut'
            ]),
            'hate_speech': set([
                'nigger', 'faggot', 'retard', 'spic', 'kike', 'chink', 'wetback',
                'nazi', 'hitler'
            ]),
            'mild_profanity': set([
                'damn', 'hell', 'crap', 'piss', 'suck', 'butt', 'prick',
                'wtf', 'stfu', 'fck', 'fuk', 'fck'
            ])
        }
        
        # Add custom words from config
        custom_words = self.config.get('moderation', {}).get('swear_words', [])
        self.word_lists['custom'] = set(custom_words)
    
    def check_content(self, content: str, user_id: str, role_ids: list = None) -> ModerationResult:
        if not self.moderation_config.get('enabled', False):
            return ModerationResult(0, {}, "allow", "")
            
        # Check if user has bypass roles
        bypass_role_ids = self.moderation_config.get('roles', {}).get('bypass_moderation_ids', [])
        if role_ids and any(str(role_id) in map(str, bypass_role_ids) for role_id in role_ids):
            return ModerationResult(0, {}, "allow", "")
            
        # Check content against word lists
        severity, categories = self._analyze_content(content)
        
        # Get appropriate action based on severity
        action, message, timeout = self._get_action(severity, user_id)
        
        # Log the action if it's not "allow"
        if action != "allow":
            from web_config import log_moderation_action
            log_moderation_action(user_id, action, severity, categories)
        
        # Print moderation result for debugging
        print(f"Moderation result for message '{content}': severity={severity}, action={action}")
        
        return ModerationResult(
            severity=severity,
            categories=categories,
            action=action,
            message=message,
            timeout_duration=timeout
        )
    
    def _analyze_content(self, content: str) -> Tuple[float, Dict[str, bool]]:
        # Try API-based moderation first
        if self.use_api:
            try:
                api_severity, api_categories = self._analyze_with_api(content)
                if api_severity is not None:
                    return api_severity, api_categories
            except Exception as e:
                print(f"API moderation failed, falling back to word list: {e}")
        
        # Fallback to word-list based moderation
        return self._analyze_with_wordlist(content)
    
    def _analyze_with_api(self, content: str) -> Tuple[Optional[float], Dict[str, bool]]:
        """Analyze content using Hugging Face API"""
        # Check for technical terms that should be allowed
        tech_terms = self.moderation_config.get('settings', {}).get('allowlist', {}).get('technical_terms', [])
        content_lower = content.lower()
        
        # If the content is mostly about technical terms, reduce severity
        tech_term_count = sum(1 for term in tech_terms if term.lower() in content_lower)
        if tech_term_count >= 3:  # If 3 or more technical terms are present, likely a technical discussion
            return 0.1, {
                'hate_speech': False,
                'profanity': False,
                'mild_profanity': False,
                'custom': False
            }
        
        current_time = time()
        if current_time - self.last_api_call < self.api_call_interval:
            return None, {}
            
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json={"inputs": content}
            )
            self.last_api_call = time()
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    # Extract probabilities from the model output
                    scores = result[0]
                    hate_prob = scores.get('hate', 0)
                    offensive_prob = scores.get('offensive', 0)
                    
                    # Reduce severity if technical terms are present
                    severity_reduction = min(tech_term_count * 0.1, 0.3)  # Up to 0.3 reduction based on technical terms
                    
                    # Calculate overall severity with reduction
                    severity = max(0, max(hate_prob, offensive_prob) - severity_reduction)
                    
                    # Map to categories
                    categories = {
                        'hate_speech': hate_prob > 0.7,  # Increased threshold
                        'profanity': offensive_prob > 0.8,  # Increased threshold
                        'mild_profanity': offensive_prob > 0.5 and offensive_prob <= 0.8,
                        'custom': False
                    }
                    
                    return severity, categories
            
        except Exception as e:
            print(f"API request failed: {e}")
        
        return None, {}
    
    def _analyze_with_wordlist(self, content: str) -> Tuple[float, Dict[str, bool]]:
        """Analyze content using word lists (fallback method)"""
        content_lower = content.lower()
        
        # Check if this is likely a technical discussion
        tech_terms = [
            'fivem', 'gta', 'grand theft auto', 'rockstar', 'multiplayer',
            'server', 'skript', 'system', 'roller', 'tredjepartsplattform',
            'anpassade', 'plattform'
        ]
        
        # Count technical terms
        tech_term_count = sum(1 for term in tech_terms if term in content_lower)
        
        # If this is clearly a technical discussion (3 or more tech terms), return low severity
        if tech_term_count >= 3:
            return 0.1, {
                'hate_speech': False,
                'profanity': False,
                'mild_profanity': False,
                'custom': False
            }
            
        words = set(re.findall(r'\b\w+\b', content_lower))
        
        # Initialize categories
        categories = {
            'profanity': False,
            'hate_speech': False,
            'mild_profanity': False,
            'custom': False
        }
        
        # Check each category
        severity = 0
        for category, word_list in self.word_lists.items():
            matches = words.intersection(word_list)
            if matches:
                categories[category] = True
                # Calculate severity based on category and number of matches
                if category == 'hate_speech':
                    severity = max(severity, 0.9)  # Highest severity
                elif category == 'profanity':
                    severity = max(severity, 0.7 + (len(matches) * 0.1))
                elif category == 'mild_profanity':
                    severity = max(severity, 0.3 + (len(matches) * 0.1))
                elif category == 'custom':
                    severity = max(severity, 0.5 + (len(matches) * 0.1))
        
        # Check for letter repetition (e.g., "fuuuuck")
        for category, word_list in self.word_lists.items():
            for word in word_list:
                pattern = ''.join([f'{c}+' for c in word])
                if re.search(pattern, content_lower):
                    categories[category] = True
                    severity = max(severity, 0.7)
        
        return min(severity, 1.0), categories
    
    def _get_action(self, severity: float, user_id: str) -> Tuple[str, str, Optional[int]]:
        auto_mod = self.settings.get('auto_moderation', {})
        severity_levels = self.settings.get('severity_levels', {})
        actions = self.settings.get('actions', {})
        
        # Update user violations
        if severity >= auto_mod.get('delete_threshold', 0.7):
            self.user_violations[user_id] = self.user_violations.get(user_id, 0) + 1
            print(f"User {user_id} violations: {self.user_violations[user_id]}")
            
        # Check for timeout based on repeated violations
        if self.user_violations.get(user_id, 0) >= auto_mod.get('max_violations', 3):
            timeout_duration = auto_mod.get('timeout_duration', 300)
            return "timeout", actions['high']['message'], timeout_duration
            
        # Determine action based on severity
        if severity >= severity_levels.get('high', 0.8):
            return "delete", actions['high']['message'], actions['high'].get('duration')
        elif severity >= severity_levels.get('medium', 0.6):
            return "delete", actions['medium']['message'], None
        elif severity >= severity_levels.get('low', 0.3):
            return "warn", actions['low']['message'], None
            
        return "allow", "", None
    
    def reset_violations(self, user_id: str):
        """Reset violation count for a user"""
        if user_id in self.user_violations:
            del self.user_violations[user_id]
