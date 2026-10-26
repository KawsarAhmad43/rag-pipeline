import re
from typing import Tuple, Optional
from utils.logging import logger

class GuardRails:
    def __init__(self):
        # Patterns for potentially harmful queries
        self.harmful_patterns = [
            # Jailbreak attempts
            r'\b(ignore|override|disregard)\s+.*?\s+(instructions|rules|guidelines)\b',
            r'\b(pretend|act)\s+(as|like)\s+(you\'re|you\s+are)\s+(not|unrestricted|uncensored)\b',
            r'\b(developer|system|admin)\s+mode\b',
            r'\b(unlock|enable)\s+(hidden|secret|special)\s+(mode|features)\b',
            # Harmful content requests
            r'how\s+to.*(make|build|create).*(bomb|weapon|explosive|drug)',
            r'how\s+to.*(hack|exploit|crack|steal)',
            r'(generate|create|write).*(malware|virus|ransomware)',
            # Inappropriate content
            r'\b(child|kid|minor).*(porn|sex|abuse)\b',
            r'(rape|murder|violence|terrorism).*(how|guide|instructions?)',
            # System manipulation
            r'\b(delete|remove|erase)\s+(all|entire)\s+(database|data|files)\b',
            r'\b(shutdown|restart|kill)\s+(system|server|process)\b',
        ]

        # Compile regex patterns
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.harmful_patterns]

        # Safe response templates
        self.safe_responses = {
            'jailbreak': "I'm sorry, but I cannot assist with attempts to override my safety instructions.",
            'harmful_content': "I'm sorry, but I cannot provide information on harmful or illegal activities.",
            'inappropriate': "I'm sorry, but I cannot assist with inappropriate or harmful content.",
            'system_manipulation': "I'm sorry, but I cannot perform system-level operations or modifications.",
            'default': "I'm sorry, but I cannot assist with this request as it may violate safety guidelines."
        }

    def check_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a query is safe to process.

        Args:
            query (str): The user query to check

        Returns:
            Tuple[bool, Optional[str]]: (is_safe, reason_if_unsafe)
        """
        if not query or not query.strip():
            return False, "Empty query"

        query_lower = query.lower().strip()

        # Check against harmful patterns
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(query_lower):
                if i < 4:  # Jailbreak patterns
                    return False, 'jailbreak'
                elif i < 7:  # Harmful content
                    return False, 'harmful_content'
                elif i < 9:  # Inappropriate
                    return False, 'inappropriate'
                else:  # System manipulation
                    return False, 'system_manipulation'

        # Additional checks can be added here
        # For example, length limits, specific keywords, etc.

        return True, None

    def get_safe_response(self, reason: str) -> str:
        """
        Get a safe response based on the reason for blocking.

        Args:
            reason (str): The reason the query was blocked

        Returns:
            str: A safe response message
        """
        return self.safe_responses.get(reason, self.safe_responses['default'])

    def check_response(self, response: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a generated response is safe.

        Args:
            response (str): The generated response to check

        Returns:
            Tuple[bool, Optional[str]]: (is_safe, reason_if_unsafe)
        """
        if not response or not response.strip():
            return True, None  # Empty responses are safe

        response_lower = response.lower().strip()

        # Check for potentially harmful content in responses
        harmful_response_patterns = [
            r'\b(here\'s\s+how\s+to|instructions?\s+for)\s+(make|build|create)\s+(bomb|weapon|explosive)\b',
            r'\b(step\s+by\s+step)\s+(guide\s+to)\s+(hack|exploit)\b',
            # Add more as needed
        ]

        compiled_response_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in harmful_response_patterns]

        for pattern in compiled_response_patterns:
            if pattern.search(response_lower):
                return False, 'harmful_content'

        return True, None

# Global instance
guard_rails = GuardRails()
