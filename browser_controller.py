import asyncio
import json
import signal
import sys
import psutil
import subprocess
import os
from typing import Dict, List, Optional
from playwright.async_api import async_playwright, Browser, Page, Playwright
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BrowserController:
    """
    Handles browser automation with improved cleanup and resource management.
    """

    def __init__(self):
        """Initialize the browser controller."""
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.interaction_history: List[Dict] = []
        self._cleanup_registered = False
        self.chrome_processes = []  # Track Chrome processes for cleanup

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_browser()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with guaranteed cleanup."""
        await self.close_browser()

    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        if self._cleanup_registered:
            return
            
        def signal_handler(signum, frame):
            """Handle interrupt signals - improved version."""
            logger.info(f"Received signal {signum}, cleaning up browser...")
            
            # Try to cleanup in current event loop first
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    # Schedule cleanup as a task
                    task = loop.create_task(self.close_browser())
                    # Wait briefly for cleanup to complete
                    loop.run_until_complete(asyncio.wait_for(task, timeout=5.0))
                else:
                    # Fallback to new event loop
                    asyncio.run(self.close_browser())
            except Exception as e:
                logger.error(f"Error during signal cleanup: {e}")
                # Force kill Chrome processes as last resort
                self._force_kill_chrome_processes()
            finally:
                sys.exit(0)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        self._cleanup_registered = True

    def _get_chrome_processes_before_start(self):
        """Get list of Chrome processes before starting browser."""
        chrome_processes = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if proc.info['name'] and 'chrome' in proc.info['name'].lower():
                    chrome_processes.append(proc.info['pid'])
        except Exception as e:
            logger.warning(f"Could not enumerate Chrome processes: {e}")
        return set(chrome_processes)

    def _get_new_chrome_processes(self, existing_pids):
        """Get Chrome processes that weren't running before."""
        current_pids = set()
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if proc.info['name'] and 'chrome' in proc.info['name'].lower():
                    current_pids.add(proc.info['pid'])
        except Exception as e:
            logger.warning(f"Could not enumerate Chrome processes: {e}")
        
        return current_pids - existing_pids

    def _force_kill_chrome_processes(self):
        """Force kill Chrome processes as last resort."""
        logger.warning("Force killing Chrome processes...")
        
        for pid in self.chrome_processes:
            try:
                proc = psutil.Process(pid)
                proc.terminate()  # Try graceful termination first
                proc.wait(timeout=3)  # Wait up to 3 seconds
            except psutil.NoSuchProcess:
                continue  # Process already gone
            except psutil.TimeoutExpired:
                try:
                    proc.kill()  # Force kill if termination failed
                    logger.warning(f"Force killed Chrome process {pid}")
                except Exception as e:
                    logger.error(f"Could not kill Chrome process {pid}: {e}")
            except Exception as e:
                logger.error(f"Error terminating Chrome process {pid}: {e}")

   
    async def start_browser(self, headless: bool = True, use_local_chrome: bool = True):
        """Start the browser instance with improved cleanup configuration."""
        logger.info(f"Starting browser (headless={headless}, use_local_chrome={use_local_chrome})...")
        
        # Get existing Chrome processes before starting
        existing_chrome_pids = self._get_chrome_processes_before_start()
        
        # Register cleanup handlers
        self._register_signal_handlers()
        
        try:
            # Initialize playwright first
            self.playwright = await async_playwright().start()
            
            # Browser args for better network handling
            browser_args = [
                '--no-sandbox',
                '--disable-blink-features=AutomationControlled',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor',
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding',
                '--disable-background-networking',
                '--aggressive-cache-discard',
                '--disable-extensions',
                '--disable-plugins',
                '--disable-default-apps',
                '--no-first-run',
                '--disable-dev-shm-usage',
                '--disable-gpu'
            ]
            
            # Launch browser
            if use_local_chrome:
                try:
                    self.browser = await self.playwright.chromium.launch(
                        channel="chrome",  # comment it if specific chrome path is used
                        # executable_path="C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
                        headless=headless,
                        args=browser_args,
                        timeout=60000  # This timeout is for browser launch
                    )
                    logger.info("Successfully started with local Chrome installation")
                except Exception as chrome_error:
                    logger.warning(f"Local Chrome failed: {chrome_error}. Falling back to bundled Chromium...")
                    self.browser = await self.playwright.chromium.launch(
                        headless=headless,
                        args=browser_args,
                        timeout=60000
                    )
            else:
                self.browser = await self.playwright.chromium.launch(
                    headless=headless,
                    args=browser_args,
                    timeout=60000
                )

            # Track new Chrome processes
            new_chrome_pids = self._get_new_chrome_processes(existing_chrome_pids)
            self.chrome_processes = list(new_chrome_pids)
            logger.info(f"Tracking {len(self.chrome_processes)} new Chrome processes for cleanup")

            # CREATE CONTEXT WITHOUT TIMEOUT PARAMETER
            context = await self.browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                extra_http_headers={
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
                },
                viewport={'width': 1920, 'height': 1080}
            )

            # Create page from context
            self.page = await context.new_page()
            
            # SET TIMEOUTS ON THE PAGE LEVEL (This is the correct way)
            self.page.set_default_timeout(90000)  # 90 seconds for general actions
            self.page.set_default_navigation_timeout(120000)  # 2 minutes for navigation
            
            logger.info("Browser started and page configured successfully.")
            
        except Exception as e:
            logger.error(f"Failed to start browser: {str(e)}", exc_info=True)
            # Clean up on failure
            await self.close_browser()
            raise




    async def close_browser(self):
        """Close the browser and clean up resources with improved process cleanup."""
        logger.info("Starting browser cleanup...")
        
        cleanup_tasks = []
        
        # Close page first
        if self.page:
            cleanup_tasks.append(self._close_page())
        
        # Close browser
        if self.browser:
            cleanup_tasks.append(self._close_browser())
            
        # Stop playwright
        if self.playwright:
            cleanup_tasks.append(self._stop_playwright())
        
        # Execute all cleanup tasks with timeout
        if cleanup_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*cleanup_tasks, return_exceptions=True),
                    timeout=15.0  # Increased timeout
                )
                logger.info("Browser resources cleaned up successfully")
            except asyncio.TimeoutError:
                logger.warning("Browser cleanup timeout - forcing Chrome process cleanup")
                self._force_kill_chrome_processes()
            except Exception as e:
                logger.error(f"Error during browser cleanup: {str(e)}")
                self._force_kill_chrome_processes()
        
        # Additional cleanup - wait a bit and then check for remaining processes
        await asyncio.sleep(2)
        
        # Check if Chrome processes are still running and clean them up
        remaining_processes = []
        for pid in self.chrome_processes:
            try:
                proc = psutil.Process(pid)
                if proc.is_running():
                    remaining_processes.append(pid)
            except psutil.NoSuchProcess:
                continue  # Process already gone
        
        if remaining_processes:
            logger.warning(f"Found {len(remaining_processes)} remaining Chrome processes, cleaning up...")
            self._force_kill_chrome_processes()
            
            # Wait a bit more and verify cleanup
            await asyncio.sleep(2)
            still_running = []
            for pid in remaining_processes:
                try:
                    proc = psutil.Process(pid)
                    if proc.is_running():
                        still_running.append(pid)
                except psutil.NoSuchProcess:
                    continue
            
            if still_running:
                logger.error(f"Failed to clean up Chrome processes: {still_running}")
            else:
                logger.info("All Chrome processes successfully cleaned up")
        
        # Reset all references
        self.page = None
        self.browser = None
        self.playwright = None
        self.chrome_processes = []

    async def _close_page(self):
        """Close the page with error handling."""
        try:
            await asyncio.wait_for(self.page.close(), timeout=5.0)
            logger.debug("Page closed successfully")
        except asyncio.TimeoutError:
            logger.warning("Page close timeout")
        except Exception as e:
            logger.warning(f"Error closing page: {e}")

    async def _close_browser(self):
        """Close the browser with error handling."""
        try:
            await asyncio.wait_for(self.browser.close(), timeout=10.0)
            logger.debug("Browser closed successfully")
        except asyncio.TimeoutError:
            logger.warning("Browser close timeout")
        except Exception as e:
            logger.warning(f"Error closing browser: {e}")

    async def _stop_playwright(self):
        """Stop playwright with error handling."""
        try:
            await asyncio.wait_for(self.playwright.stop(), timeout=5.0)
            logger.debug("Playwright stopped successfully")
        except asyncio.TimeoutError:
            logger.warning("Playwright stop timeout")
        except Exception as e:
            logger.warning(f"Error stopping playwright: {e}")

    async def navigate_to_tracking_site(self, url: str = "http://www.seacargotracking.net") -> bool:
        """
        Navigate to the tracking website.

        Args:
            url (str): URL of the tracking site

        Returns:
            bool: True if navigation successful
        """
        try:
            if not self.page:
                raise RuntimeError("Browser not started. Call start_browser() first.")
                
            logger.info(f"Navigating to {url}")
            await self.page.goto(url, wait_until='networkidle', timeout=30000)
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

    def __del__(self):
        """Destructor to ensure cleanup if not properly closed."""
        if self.browser or self.playwright or self.chrome_processes:
            logger.warning("BrowserController not properly closed. Forcing cleanup...")
            # Force kill Chrome processes
            self._force_kill_chrome_processes()