import asyncio
import json
from typing import Dict, List, Optional
from playwright.async_api import async_playwright, Browser, Page
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BrowserController:
    """
    Handles browser automation for shipping tracking websites.

    This class provides methods to interact with web pages using natural language
    descriptions and adaptive element detection.
    """

    def __init__(self):
        """Initialize the browser controller."""
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.interaction_history: List[Dict] = []

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_browser()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_browser()

    async def start_browser(self, headless: bool = True):
        """
        Start the browser instance.

        Args:
            headless (bool): Whether to run browser in headless mode
        """
        logger.info(f"Starting browser (headless={headless})...")
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=headless,
                args=[
                    '--no-sandbox',
                    '--disable-blink-features=AutomationControlled',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor'
                ]
            )

            # Create a new page with realistic user agent
            self.page = await self.browser.new_page()
            await self.page.set_user_agent(
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            )
            logger.info("Browser started and page created successfully.")
        except Exception as e:
            logger.error(f"Failed to start browser: {str(e)}", exc_info=True)
            raise

    async def close_browser(self):
        """Close the browser and cleanup resources."""
        logger.info("Closing browser...")
        try:
            if self.page:
                await self.page.close()
                logger.debug("Page closed.")
            if self.browser:
                await self.browser.close()
                logger.debug("Browser closed.")
            if hasattr(self, 'playwright'):
                await self.playwright.stop()
                logger.debug("Playwright stopped.")
            logger.info("Browser resources cleaned up successfully.")
        except Exception as e:
            logger.error(f"Error closing browser resources: {str(e)}", exc_info=True)

    async def navigate_to_tracking_site(self, url: str = "http://www.seacargotracking.net") -> bool:
        """
        Navigate to the tracking website.

        Args:
            url (str): URL of the tracking site

        Returns:
            bool: True if navigation successful
        """
        try:
            logger.info(f"Navigating to {url}")
            await self.page.goto(url, wait_until='networkidle')
            await self.page.wait_for_timeout(2000)  # Allow page to fully load
            logger.info(f"Successfully navigated to {url}")

            # Record the interaction
            self.interaction_history.append({
                "action": "navigate",
                "url": url,
                "timestamp": datetime.now().isoformat(),
                "success": True
            })

            return True

        except Exception as e:
            logger.error(f"Navigation to {url} failed: {str(e)}", exc_info=True)
            self.interaction_history.append({
                "action": "navigate",
                "url": url,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(e)
            })
            return False