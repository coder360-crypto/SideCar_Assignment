import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """
    Configuration management for the shipping tracking system.
    """

    def __init__(self):
        """Initialize configuration settings."""
        self.load_settings()

    def load_settings(self):
        """Load all configuration settings."""

        # API Keys
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Alternative LLM

        # Browser settings
        self.BROWSER_HEADLESS = os.getenv("BROWSER_HEADLESS", "true").lower() == "true"
        self.BROWSER_TIMEOUT = int(os.getenv("BROWSER_TIMEOUT", "30000"))
        self.BROWSER_SLOW_MO = int(os.getenv("BROWSER_SLOW_MO", "100"))

        # Database settings
        self.DATABASE_PATH = os.getenv("DATABASE_PATH", "shipping_tracking.db")
        self.CACHE_EXPIRY_HOURS = int(os.getenv("CACHE_EXPIRY_HOURS", "24"))

        # Tracking settings
        self.DEFAULT_TRACKING_URL = os.getenv(
            "DEFAULT_TRACKING_URL", 
            "http://www.seacargotracking.net"
        )
        self.MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))
        self.RETRY_DELAY_SECONDS = int(os.getenv("RETRY_DELAY_SECONDS", "5"))

        # Logging settings
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FILE = os.getenv("LOG_FILE", "shipping_tracking.log")

        # Performance settings
        self.ENABLE_SCREENSHOTS = os.getenv("ENABLE_SCREENSHOTS", "false").lower() == "true"
        self.SCREENSHOT_DIR = os.getenv("SCREENSHOT_DIR", "screenshots")

        # Validate required settings
        self.validate_config()

    def validate_config(self):
        """Validate required configuration settings."""
        required_settings = ["OPENAI_API_KEY"]
        missing_settings = []

        for setting in required_settings:
            if not getattr(self, setting):
                missing_settings.append(setting)

        if missing_settings:
            raise ValueError(
                f"Missing required configuration: {', '.join(missing_settings)}. "
                f"Please check your .env file."
            )

    def get_llm_config(self) -> Dict[str, Any]:
        """
        Get LLM configuration.

        Returns:
            Dict[str, Any]: LLM configuration parameters
        """
        return {
            "model": "gpt-4o",
            "temperature": 0.1,
            "max_tokens": 2000,
            "api_key": self.OPENAI_API_KEY
        }

    def get_browser_config(self) -> Dict[str, Any]:
        """
        Get browser configuration.

        Returns:
            Dict[str, Any]: Browser configuration parameters
        """
        return {
            "headless": self.BROWSER_HEADLESS,
            "timeout": self.BROWSER_TIMEOUT,
            "slow_mo": self.BROWSER_SLOW_MO,
            "screenshots_enabled": self.ENABLE_SCREENSHOTS,
            "screenshot_dir": self.SCREENSHOT_DIR
        }

    def get_tracking_config(self) -> Dict[str, Any]:
        """
        Get tracking configuration.

        Returns:
            Dict[str, Any]: Tracking configuration parameters
        """
        return {
            "default_url": self.DEFAULT_TRACKING_URL,
            "max_retries": self.MAX_RETRY_ATTEMPTS,
            "retry_delay": self.RETRY_DELAY_SECONDS,
            "cache_expiry_hours": self.CACHE_EXPIRY_HOURS
        }

    def __str__(self) -> str:
        """String representation of configuration (without sensitive data)."""
        safe_config = {
            "browser_headless": self.BROWSER_HEADLESS,
            "browser_timeout": self.BROWSER_TIMEOUT,
            "database_path": self.DATABASE_PATH,
            "default_tracking_url": self.DEFAULT_TRACKING_URL,
            "max_retry_attempts": self.MAX_RETRY_ATTEMPTS,
            "log_level": self.LOG_LEVEL,
            "has_openai_key": bool(self.OPENAI_API_KEY)
        }
        return f"Config({safe_config})"
