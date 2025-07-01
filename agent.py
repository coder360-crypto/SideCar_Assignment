import asyncio
import os
import sys
import signal
import subprocess
import platform
import psutil
import tempfile
import uuid
import shutil
from datetime import datetime
from typing import Dict, Optional, Tuple
import logging

# Third-party imports
from browser_use import Agent, BrowserSession
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from data_storage import DataStorage

# Load environment variables
load_dotenv()

# Global configuration for LLM provider
LLM_PROVIDER = "gemini"

# Set up basic logging configuration
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BrowserSessionManager:
    """Context manager for browser sessions to ensure proper cleanup."""
    
    def __init__(self, agent):
        self.agent = agent
        self.session = None
    
    async def __aenter__(self):
        self.session = self.agent._create_browser_session()
        return self.session
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            try:
                if hasattr(self.session, 'browser') and self.session.browser:
                    await self.session.browser.close()
                if hasattr(self.session, 'close'):
                    await self.session.close()
                logger.debug("Browser session closed via context manager")
            except Exception as e:
                logger.warning(f"Error in context manager cleanup: {e}")

class ShippingTrackingAgent:
    """
    Main AI agent for shipping line tracking with enhanced browser session management.
    """

    def __init__(self, llm_provider: str = None, api_key: str = None, 
                 use_existing_chrome: bool = False, chrome_executable_path: str = None):
        """Initialize the shipping tracking agent."""
        logger.info("Initializing ShippingTrackingAgent...")
        self.data_storage = DataStorage()
        
        # Use provided LLM provider or fall back to global setting
        self.llm_provider = (llm_provider or LLM_PROVIDER).lower()
        
        # Store the API key and browser connection settings
        self.api_key = api_key
        self.use_existing_chrome = use_existing_chrome
        self.chrome_executable_path = chrome_executable_path or self._get_default_chrome_path()
        
        # Add browser session tracking
        self._active_sessions = []
        
        # Initialize the appropriate LLM based on provider
        self.llm = self._initialize_llm()
        
        # Register signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        logger.info(f"ShippingTrackingAgent initialized with {self.llm_provider.upper()} LLM.")

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, cleaning up...")
            asyncio.create_task(self._cleanup_browser_sessions())
            asyncio.create_task(self._force_kill_chrome_processes())
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _get_default_chrome_path(self):
        """Get the default Chrome executable path based on the operating system."""
        system = platform.system()
        
        if system == "Darwin":  # macOS
            return "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        elif system == "Windows":
            return "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
        elif system == "Linux":
            return "/usr/bin/google-chrome"
        else:
            return None

    async def _ensure_clean_start(self):
        """Ensure no conflicting Chrome processes are running before starting."""
        try:
            # Force cleanup any existing processes
            await self._force_kill_chrome_processes()
            
            # Clean up any temporary profile directories
            temp_dir = tempfile.gettempdir()
            for item in os.listdir(temp_dir):
                if item.startswith("browseruse_profile_"):
                    profile_path = os.path.join(temp_dir, item)
                    try:
                        shutil.rmtree(profile_path, ignore_errors=True)
                    except:
                        pass
            
            logger.info("Clean start ensured")
        except Exception as e:
            logger.warning(f"Error ensuring clean start: {e}")

    def _create_browser_session(self):
        """Create a browser session configuration for the Agent."""
        if self.use_existing_chrome and self.chrome_executable_path:
            logger.info(f"Using Chrome executable at: {self.chrome_executable_path}")
            try:
                # Create unique profile directory for each session
                profile_dir = os.path.join(tempfile.gettempdir(), f"browseruse_profile_{uuid.uuid4().hex[:8]}")
                
                session = BrowserSession(
                    executable_path=self.chrome_executable_path,
                    headless=False,
                    user_data_dir=profile_dir,  # CHANGED: Use unique profile directory
                    ignore_https_errors=True,
                    timeout=30000,
                    args=[
                        '--ignore-certificate-errors',
                        '--ignore-ssl-errors',
                        '--ignore-certificate-errors-spki-list',
                        '--disable-web-security',
                        '--allow-running-insecure-content',
                        '--no-first-run',
                        '--disable-default-apps',
                        '--disable-popup-blocking',
                        '--disable-background-timer-throttling',  # ADDED: Prevent background throttling
                        '--disable-backgrounding-occluded-windows',  # ADDED
                        '--disable-renderer-backgrounding'  # ADDED
                    ]
                )
                self._active_sessions.append(session)
                return session
            except Exception as e:
                logger.warning(f"Failed to create Chrome browser session: {e}. Using default browser.")
                return None
        else:
            logger.info("Using default Playwright browser session")
            try:
                session = BrowserSession(
                    headless=False,
                    ignore_https_errors=True,
                    timeout=30000
                )
                self._active_sessions.append(session)
                return session
            except Exception as e:
                logger.warning(f"Failed to create browser session: {e}")
                return None

    async def _cleanup_browser_sessions(self):
        """Enhanced cleanup for all active browser sessions."""
        logger.info(f"Cleaning up {len(self._active_sessions)} browser sessions...")
        for session in self._active_sessions:
            try:
                # Try multiple cleanup approaches
                if hasattr(session, 'browser') and session.browser:
                    await session.browser.close()
                    logger.debug("Browser instance closed")
                
                if hasattr(session, 'close'):
                    await session.close()
                    logger.debug("Session wrapper closed")
                    
                # Force cleanup of any remaining processes
                if hasattr(session, 'playwright_browser'):
                    await session.playwright_browser.close()
                    
            except Exception as e:
                logger.warning(f"Error closing browser session: {e}")
        
        self._active_sessions.clear()
        logger.info("All browser sessions cleaned up")

    async def _force_kill_chrome_processes(self):
        """Force kill any remaining Chrome processes."""
        try:
            system = platform.system()
            
            if system == "Windows":
                # ENHANCED: Kill both chrome.exe and chromedriver.exe
                subprocess.run(["taskkill", "/f", "/im", "chrome.exe"], 
                             capture_output=True, check=False)
                subprocess.run(["taskkill", "/f", "/im", "chromedriver.exe"], 
                             capture_output=True, check=False)
            elif system in ["Darwin", "Linux"]:
                # ENHANCED: Kill chrome processes more specifically
                subprocess.run(["pkill", "-f", "chrome"], 
                             capture_output=True, check=False)
                subprocess.run(["pkill", "-f", "chromedriver"], 
                             capture_output=True, check=False)
            
            # ADDED: Wait for processes to terminate
            await asyncio.sleep(2)
            logger.info("Force killed Chrome processes")
        except Exception as e:
            logger.warning(f"Could not force kill Chrome processes: {e}")

    def _initialize_llm(self):
        """Initialize the appropriate LLM based on the provider setting."""
        if self.llm_provider == "gemini":
            logger.info("Initializing Gemini LLM...")
            api_key = self.api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable or api_key parameter is required for Gemini")
            
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-preview-05-20",
                temperature=1.0,
                google_api_key=api_key
            )
            
        elif self.llm_provider == "openai":
            logger.info("Initializing OpenAI LLM...")
            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable or api_key parameter is required for OpenAI")
            
            return ChatOpenAI(
                model="o4-mini-2025-04-16",
                api_key=api_key
            )
        
        elif self.llm_provider == "groq":
            logger.info("Initializing Groq LLM...")
            api_key = self.api_key or os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable or api_key parameter is required for Groq")
            
            return ChatGroq(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0.2,
                api_key=api_key
            )
            
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def get_llm_info(self) -> Dict[str, str]:
        """Get information about the current LLM configuration."""
        if self.llm_provider == "gemini":
            return {
                "provider": "Google Gemini",
                "model": "gemini-2.5-flash-preview-05-20",
                "api_key_set": bool(self.api_key or os.getenv("GOOGLE_API_KEY"))
            }
        elif self.llm_provider == "openai":
            return {
                "provider": "OpenAI",
                "model": "o4-mini-2025-04-16", 
                "api_key_set": bool(self.api_key or os.getenv("OPENAI_API_KEY"))
            }
        elif self.llm_provider == "groq":
            return {
                "provider": "Groq",
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "api_key_set": bool(self.api_key or os.getenv("GROQ_API_KEY"))
            }

    async def track_shipment(self, booking_id: str) -> Dict:
        """Track a shipment using the provided booking ID."""
        print(f"\n🚢 Starting shipment tracking for booking ID: {booking_id}")
        print(f"🤖 Using {self.get_llm_info()['provider']} ({self.get_llm_info()['model']})")
        logger.info(f"Starting shipment tracking for booking ID: {booking_id} with {self.llm_provider}")

        try:
            # ADDED: Ensure clean start before beginning
            await self._ensure_clean_start()
            
            # Step 1: Check if we have cached successful patterns
            logger.debug(f"Checking cache for booking ID: {booking_id}")
            cached_result = await self.data_storage.get_cached_result(booking_id)
            if cached_result:
                print("✅ Found cached result!")
                logger.info(f"Found cached result for booking ID: {booking_id}")
                return cached_result

            # Step 2: Use AI to navigate and extract information
            logger.debug(f"Performing AI tracking for booking ID: {booking_id}")
            result = await self._perform_ai_tracking(booking_id)

            # Step 3: Store the successful interaction pattern
            if "error" not in result:
                logger.debug(f"Storing interaction pattern for booking ID: {booking_id}")
                await self.data_storage.store_interaction_pattern(booking_id, result)
                print("✅ Successfully retrieved shipping information!")
                logger.info(f"Successfully retrieved shipping information for booking ID: {booking_id}")
            else:
                logger.warning(f"Skipping storing pattern due to error for booking ID: {booking_id}")
            
            return result

        except Exception as e:
            print(f"❌ Error tracking shipment: {str(e)}")
            logger.error(f"Error tracking shipment for booking ID {booking_id}: {str(e)}", exc_info=True)
            return {"error": str(e), "booking_id": booking_id}
        
        finally:
            # Cleanup any remaining browser sessions
            await self._cleanup_browser_sessions()

    async def _perform_ai_tracking(self, booking_id: str) -> Dict:
        """Perform AI-driven tracking using natural language instructions with proper session management."""
        
        # Natural language prompt for the AI agent
        tracking_prompt = f"""
You are a web automation agent tasked with tracking shipping container/booking ID '{booking_id}' using HMM (Hyundai Merchant Marine) on seacargotracking.net.

EXECUTION STRATEGY:
- Execute each step completely before moving to the next
- Wait for page loads between actions (minimum 3 seconds)
- Take screenshots after each major action for verification
- If any step fails, document the failure and attempt alternative approaches

STEP 1: INITIAL NAVIGATION
1. Navigate to http://www.seacargotracking.net
2. Wait for complete page load (check for loading indicators)
3. Take screenshot and document page structure
4. DO NOT interact with any tracking elements on the main page

STEP 2: LOCATE HMM WITH SYSTEMATIC SEARCH
5. Scroll down in 200-pixel increments (not 100) to avoid missing content
6. After each scroll, pause 2 seconds and scan for:
   - Text containing "HMM", "Hyundai Merchant Marine", or "Hyundai"
   - Logo images with HMM branding
   - Links or buttons with HMM references
7. Check these specific areas:
   - Shipping line directory/list
   - Partner logos section
   - Footer links
   - Navigation menus
8. STOP scrolling immediately when HMM is found - document exact location
9. If not found after reaching page bottom, check for "Load More" or pagination

STEP 3: ACCESS HMM TRACKING
10. Click on the HMM link/logo/button you found
11. Wait 5 seconds for navigation to complete
12. Verify you're on HMM-specific page (check URL and page title)
13. Look for "Container Tracking", "Track & Trace", or similar sections
14. If multiple tracking options exist, prioritize "Container Tracking"

STEP 4: EXECUTE TRACKING WITH VALIDATION
15. Locate the tracking input form
16. Identify field labels carefully:
    - "Container Number" or "Container No"
    - "Booking Number" or "Booking Reference" 
    - "Bill of Lading" or "B/L Number"
17. Enter '{booking_id}' in the most appropriate field based on the ID format
18. Click the tracking/search button
19. Wait minimum 10 seconds for results
20. Check for loading indicators and wait until they disappear

STEP 5: RESULT PROCESSING AND FALLBACK
21. Examine the results page completely:
    - Look for data tables, status information, or tracking timelines
    - Check for error messages like "No results found" or "Invalid number"
22. If tracking fails:
    - Try removing prefixes from booking_id (keep only numbers)
    - Try different input fields if available
    - Look for format requirements or examples on the page
23. If still no results, try alternative tracking methods on the same page

ENHANCED ERROR HANDLING:
- If HMM is not found: Document all shipping lines you did find
- If tracking page doesn't load: Try refreshing or alternative navigation
- If no tracking form exists: Document what tracking options are available
- If results are unclear: Screenshot and describe exactly what you see

SUCCESS CRITERIA:
- Must show actual container/shipment data (not just "search completed")
- Must display at least 2 of: status, vessel info, port info, or dates
- "Not specified" or empty fields don't count as successful tracking

RESPONSE FORMAT:
{{
    "tracking_status": "SUCCESS" or "FAILED",
    "booking_id": "{booking_id}",
    "execution_log": [
        "Step 1: Navigated to main page - loaded successfully",
        "Step 2: Found HMM in shipping lines section after 3 scrolls",
        "Step 3: Clicked HMM, loaded tracking page",
        "Step 4: Entered booking ID in container field, clicked search",
        "Step 5: Results loaded showing container status"
    ],
    "hmm_location": "exact description of where HMM was found",
    "error_details": "specific error messages or issues encountered",
    "page_screenshots": "descriptions of key screenshots taken",
    "container_details": {{
        "status": "current container status",
        "vessel_name": "vessel name if available",
        "voyage_number": "voyage number if available", 
        "port_of_loading": "departure port",
        "port_of_discharge": "destination port",
        "departure_date": "departure date if available",
        "arrival_date": "arrival date if available",
        "current_location": "current position/location",
        "milestones": ["chronological list of tracking events"],
        "delivery_status": "delivery status if available",
        "additional_info": "any other relevant tracking data found"
    }}
}}
"""


        logger.debug(f"Performing AI tracking for {booking_id}. Prompt: {tracking_prompt[:200]}...")

        # Use context manager for proper session cleanup
        async with BrowserSessionManager(self) as browser_session:
            try:
                # Create agent with or without custom browser session
                if browser_session:
                    tracking_agent = Agent(
                        task=tracking_prompt,
                        llm=self.llm,
                        browser_session=browser_session
                    )
                else:
                    tracking_agent = Agent(
                        task=tracking_prompt,
                        llm=self.llm
                    )

                # Execute the AI-driven browser automation
                try:
                    logger.info(f"Running BrowserAgent for booking ID: {booking_id} with max_steps=10")
                    result = await tracking_agent.run(max_steps=10)
                    logger.debug(f"BrowserAgent raw result for {booking_id}: {str(result)[:200]}...")

                    # Parse and structure the result
                    logger.debug(f"Parsing tracking result for {booking_id}")
                    structured_result = await self._parse_tracking_result(result, booking_id)
                    logger.info(f"Successfully performed AI tracking for {booking_id}")
                    return structured_result

                except Exception as e:
                    print(f"Primary tracking failed: {str(e)}")
                    logger.error(f"Primary AI tracking failed for {booking_id}: {str(e)}", exc_info=True)
                    # Fallback: try alternative approach
                    return await self._fallback_tracking_method(booking_id)

            except Exception as e:
                logger.error(f"Error in browser session management for {booking_id}: {e}")
                return {
                    "booking_id": booking_id,
                    "error": f"Browser session error: {str(e)}",
                    "retrieved_at": datetime.now().isoformat(),
                    "llm_provider": self.llm_provider
                }

    async def _parse_tracking_result(self, agent_history, booking_id: str) -> Dict:
        """Parse the result from AI browser automation into structured data."""
        
        # Extract text content from agent history
        try:
            logger.debug(f"Extracting text from agent history for {booking_id}")
            if hasattr(agent_history, 'final_result'):
                raw_result = str(agent_history.final_result)
            elif hasattr(agent_history, '__iter__'):
                raw_result = ""
                for step in agent_history:
                    if hasattr(step, 'result') and step.result:
                        raw_result += str(step.result) + "\n"
            else:
                raw_result = str(agent_history)
            logger.debug(f"Extracted raw_result for LLM parsing: {raw_result[:200]}...")
                
        except Exception as e:
            logger.warning(f"Could not extract text from agent_history: {e}", exc_info=True)
            raw_result = str(agent_history)

        # Use LLM to extract structured information
        extraction_prompt = f"""
        Extract shipping tracking information from the following text and return it as JSON:

        Text: {raw_result}

        Extract:
        - voyage_number: The voyage or vessel number
        - arrival_date: The expected arrival date (format: YYYY-MM-DD if possible)
        - departure_date: The departure date if available
        - vessel_name: The name of the vessel/ship
        - port_of_loading: Port where cargo was loaded
        - port_of_discharge: Port where cargo will be discharged
        - status: Current status of the shipment

        Return as valid JSON only, no additional text.
        """

        logger.debug(f"Parsing raw result with {self.llm_provider.upper()} LLM for {booking_id}")

        try:
            logger.info(f"Invoking {self.llm_provider.upper()} LLM for data extraction for booking ID: {booking_id}")
            response = self.llm.invoke(extraction_prompt)
            extracted_data = response.content.strip()
            logger.debug(f"{self.llm_provider.upper()} LLM raw extracted_data for {booking_id}: {extracted_data[:200]}...")

            # Clean the response to ensure it's valid JSON
            if extracted_data.startswith('```'):
                extracted_data = extracted_data.replace('```json', '').replace('```', '')

            # Parse JSON response
            import json
            parsed_data = json.loads(extracted_data)

            # Add metadata
            parsed_data.update({
                "booking_id": booking_id,
                "retrieved_at": datetime.now().isoformat(),
                "source": "seacargotracking.net",
                "method": "ai_browser_automation",
                "llm_provider": self.llm_provider
            })

            logger.debug(f"Successfully parsed {self.llm_provider.upper()} LLM response for {booking_id}")
            return parsed_data

        except Exception as e:
            print(f"Failed to parse result: {str(e)}")
            logger.error(f"Failed to parse {self.llm_provider.upper()} LLM result for {booking_id}: {str(e)}")
            return {
                "booking_id": booking_id,
                "error": f"Failed to parse result: {str(e)}",
                "raw_result": str(raw_result),
                "retrieved_at": datetime.now().isoformat(),
                "llm_provider": self.llm_provider
            }

    async def _fallback_tracking_method(self, booking_id: str) -> Dict:
        """Fallback method when primary AI tracking fails."""
        
        print("🔄 Attempting fallback tracking method...")
        logger.info(f"Attempting fallback tracking method for booking ID: {booking_id}")

        fallback_prompt = f"""
        The primary tracking method failed. Try these alternative approaches:

        1. Search for '{booking_id}' on HMM's official website (hmm21.com)
        2. Try other container tracking sites that support HMM
        3. Look for any alternative tracking interfaces or APIs

        Booking ID to track: {booking_id}

        Extract voyage number and arrival date information.
        """
        
        logger.debug(f"Fallback prompt for {booking_id}: {fallback_prompt[:200]}...")
        
        # Use context manager for fallback session
        async with BrowserSessionManager(self) as browser_session:
            try:
                if browser_session:
                    fallback_agent = Agent(
                        task=fallback_prompt,
                        llm=self.llm,
                        browser_session=browser_session
                    )
                else:
                    fallback_agent = Agent(
                        task=fallback_prompt,
                        llm=self.llm
                    )

                logger.info(f"Running fallback BrowserAgent for {booking_id} with max_steps=8")
                result = await fallback_agent.run(max_steps=8)
                logger.debug(f"Fallback BrowserAgent raw result for {booking_id}: {str(result)[:200]}...")
                return await self._parse_tracking_result(result, booking_id)
                
            except Exception as e:
                print(f"Fallback method also failed: {str(e)}")
                logger.error(f"Fallback tracking method failed for {booking_id}: {str(e)}", exc_info=True)
                return {
                    "booking_id": booking_id,
                    "error": f"All tracking methods failed: {str(e)}",
                    "retrieved_at": datetime.now().isoformat(),
                    "llm_provider": self.llm_provider
                }

    async def track_multiple_shipments(self, booking_ids: list) -> Dict[str, Dict]:
        """Track multiple shipments concurrently."""
        print(f"\n🚢 Starting batch tracking for {len(booking_ids)} shipments...")
        print(f"🤖 Using {self.get_llm_info()['provider']} ({self.get_llm_info()['model']})")
        logger.info(f"Starting batch tracking for {len(booking_ids)} shipments: {', '.join(booking_ids)} with {self.llm_provider}")
        
        tasks = [self.track_shipment(booking_id) for booking_id in booking_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            booking_ids[i]: results[i] if not isinstance(results[i], Exception) 
            else {"error": str(results[i]), "booking_id": booking_ids[i], "llm_provider": self.llm_provider}
            for i in range(len(booking_ids))
        }

async def main():
    """Main function to run the shipping tracking agent from command line."""
    if len(sys.argv) != 2:
        print("Usage: python agent.py <booking_id>")
        print("Example: python agent.py SINI25432400")
        sys.exit(1)
    
    booking_id = sys.argv[1]
    agent = None
    
    try:
        # ADDED: Quick cleanup before starting
        system = platform.system()
        if system == "Windows":
            subprocess.run(["taskkill", "/f", "/im", "chrome.exe"], 
                         capture_output=True, check=False)
        elif system in ["Darwin", "Linux"]:
            subprocess.run(["pkill", "-f", "chrome"], 
                         capture_output=True, check=False)
        await asyncio.sleep(2)  # Wait for processes to terminate
        
        # Create the tracking agent with Chrome support
        agent = ShippingTrackingAgent(use_existing_chrome=True)
        
        # Track the shipment
        result = await agent.track_shipment(booking_id)
        
        # Print the results in a formatted way
        print("\n" + "="*60)
        print("📦 SHIPPING TRACKING RESULTS")
        print("="*60)
        
        if "error" in result:
            print(f"❌ Tracking failed for booking ID: {booking_id}")
            print(f"Error: {result['error']}")
            if "raw_result" in result:
                print(f"Raw result: {result['raw_result'][:500]}...")
        else:
            print(f"✅ Tracking successful for booking ID: {booking_id}")
            print(f"🤖 LLM Provider: {result.get('llm_provider', 'Unknown')}")
            print(f"📅 Retrieved at: {result.get('retrieved_at', 'Unknown')}")
            print(f"🌐 Source: {result.get('source', 'Unknown')}")
            print()
            
            # Print container details if available
            if "container_details" in result:
                details = result["container_details"]
                print("📋 CONTAINER DETAILS:")
                print(f"   Status: {details.get('status', 'N/A')}")
                print(f"   Vessel: {details.get('vessel_name', 'N/A')}")
                print(f"   Voyage: {details.get('voyage_number', 'N/A')}")
                print(f"   Port of Loading: {details.get('port_of_loading', 'N/A')}")
                print(f"   Port of Discharge: {details.get('port_of_discharge', 'N/A')}")
                print(f"   Departure Date: {details.get('departure_date', 'N/A')}")
                print(f"   Arrival Date: {details.get('arrival_date', 'N/A')}")
                print(f"   Current Location: {details.get('current_location', 'N/A')}")
                print(f"   Delivery Status: {details.get('delivery_status', 'N/A')}")
                
                if details.get('milestones'):
                    print(f"   Milestones: {', '.join(details['milestones'])}")
            else:
                # Print direct fields if container_details not present
                print("📋 TRACKING INFORMATION:")
                print(f"   Voyage Number: {result.get('voyage_number', 'N/A')}")
                print(f"   Arrival Date: {result.get('arrival_date', 'N/A')}")
                print(f"   Departure Date: {result.get('departure_date', 'N/A')}")
                print(f"   Vessel Name: {result.get('vessel_name', 'N/A')}")
                print(f"   Port of Loading: {result.get('port_of_loading', 'N/A')}")
                print(f"   Port of Discharge: {result.get('port_of_discharge', 'N/A')}")
                print(f"   Status: {result.get('status', 'N/A')}")
            
            # Print exploration log if available
            if "exploration_log" in result:
                print(f"\n🔍 EXPLORATION LOG:")
                print(result["exploration_log"])
        
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        logger.info("Process interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        logger.error(f"Fatal error in main: {str(e)}", exc_info=True)
    finally:
        if agent:
            try:
                await agent._cleanup_browser_sessions()
                await agent._force_kill_chrome_processes()
                logger.info("Final cleanup completed")
            except Exception as e:
                logger.warning(f"Error during final cleanup: {e}")

# Run with proper event loop handling
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Process terminated")
    finally:
        # Final cleanup attempt
        try:
            system = platform.system()
            if system == "Windows":
                subprocess.run(["taskkill", "/f", "/im", "chrome.exe"], 
                             capture_output=True, check=False)
            elif system in ["Darwin", "Linux"]:
                subprocess.run(["pkill", "-f", "chrome"], 
                             capture_output=True, check=False)
        except:
            pass