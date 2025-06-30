
import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, Optional, Tuple
import logging
import concurrent.futures
import threading

# Third-party imports
from browser_use import Agent, BrowserSession, browser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Local imports
from browser_controller import BrowserController
from data_storage import DataStorage

# Load environment variables
load_dotenv()

# Global configuration for LLM provider
LLM_PROVIDER = "openai"  # Changed default to OpenAI for better async support

# Set up basic logging configuration
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ShippingTrackingAgent:
    """
    Main AI agent for shipping line tracking with integrated browser controller.
    
    This agent uses natural language processing to interact with websites
    and extract shipping information without hardcoded interactions.
    """

    def __init__(self, llm_provider: str = None, api_key: str = None, headless: bool = True):
        """
        Initialize the shipping tracking agent with browser controller.
        
        Args:
            llm_provider (str, optional): Override the global LLM provider
            api_key (str, optional): API key for the chosen LLM provider
            headless (bool): Whether to run browser in headless mode
        """
        logger.info("Initializing ShippingTrackingAgent with BrowserController...")
        
        self.data_storage = DataStorage()
        self.browser_controller = BrowserController()
        self.headless = headless
        
        # LLM configuration
        self.llm_provider = (llm_provider or LLM_PROVIDER).lower()
        self.api_key = api_key
        self.llm = self._initialize_llm()
        
        # Browser session will be created when needed
        self.browser_session = None
        
        logger.info(f"ShippingTrackingAgent initialized with {self.llm_provider.upper()} LLM.")

    def _initialize_llm(self):
        """
        Initialize the appropriate LLM based on the provider setting.
        
        Returns:
            LLM instance (ChatOpenAI, ChatGoogleGenerativeAI, or ChatGroq)
        """
        if self.llm_provider == "gemini":
            logger.info("Initializing Gemini LLM...")
            logger.warning("Note: Gemini LLM has limited async support. Using synchronous execution.")
            
            api_key = self.api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable or api_key parameter is required for Gemini")
            
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-preview-05-20",  # Updated to stable model
                temperature=0.5,
                google_api_key=api_key,
                transport="rest"  # Use REST transport for better compatibility
            )
            
        elif self.llm_provider == "openai":
            logger.info("Initializing OpenAI LLM...")
            
            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable or api_key parameter is required for OpenAI")
            
            return ChatOpenAI(
                model="gpt-4o-mini",  # Updated to current model
                api_key=api_key,
                temperature=0.5
            )
        
        elif self.llm_provider == "groq":
            logger.info("Initializing Groq LLM...")
            
            api_key = self.api_key or os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable or api_key parameter is required for Groq")
            
            return ChatGroq(
                model="llama-3.1-70b-versatile",  # Updated to stable model
                temperature=0.2,
                api_key=api_key
            )
            
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}. Supported providers: 'openai', 'gemini', 'groq'")

    def get_llm_info(self) -> Dict[str, str]:
        """
        Get information about the current LLM configuration.
        
        Returns:
            Dict containing LLM provider and model information
        """
        if self.llm_provider == "gemini":
            return {
                "provider": "Google Gemini",
                "model": "gemini-2.5-flash-preview-05-20",
                "api_key_set": bool(self.api_key or os.getenv("GOOGLE_API_KEY")),
                "async_support": "Limited (using sync wrapper)"
            }
        elif self.llm_provider == "openai":
            return {
                "provider": "OpenAI",
                "model": "gpt-4o-mini", 
                "api_key_set": bool(self.api_key or os.getenv("OPENAI_API_KEY")),
                "async_support": "Full"
            }
        elif self.llm_provider == "groq":
            return {
                "provider": "Groq",
                "model": "llama-3.1-70b-versatile",
                "api_key_set": bool(self.api_key or os.getenv("GROQ_API_KEY")),
                "async_support": "Full"
            }

    async def _setup_browser_session(self):
        """
        Set up browser session using the browser controller.
        """
        if self.browser_session is None:
            logger.info("Setting up browser session using BrowserController...")
            
            # Start the browser controller
            await self.browser_controller.start_browser(headless=self.headless)
            
            # Create BrowserSession using the playwright page from browser controller
            self.browser_session = BrowserSession(
                page=self.browser_controller.page,
                keep_alive=True  # Keep the session alive for reuse
            )
            
            logger.info("Browser session created successfully using BrowserController.")

    async def _cleanup_browser_session(self):
        """
        Clean up browser session and controller.
        """
        logger.info("Cleaning up browser session...")
        
        if self.browser_session:
            try:
                await self.browser_session.close()
            except Exception as e:
                logger.warning(f"Error closing browser session: {e}")
        
        await self.browser_controller.close_browser()
        logger.info("Browser session cleanup completed.")

    async def track_shipment(self, booking_id: str) -> Dict:
        """
        Track a shipment using the provided booking ID with browser controller.

        Args:
            booking_id (str): The booking ID to track

        Returns:
            Dict: Contains voyage_number, arrival_date, and metadata
        """
        print(f"\n🚢 Starting shipment tracking for booking ID: {booking_id}")
        print(f"🤖 Using {self.get_llm_info()['provider']} ({self.get_llm_info()['model']})")
        logger.info(f"Starting shipment tracking for booking ID: {booking_id} with {self.llm_provider}")

        try:
            # Setup browser session if not already done
            await self._setup_browser_session()
            
            # Navigate to tracking site using browser controller
            navigation_success = await self.browser_controller.navigate_to_tracking_site()
            if not navigation_success:
                return {
                    "booking_id": booking_id,
                    "error": "Failed to navigate to tracking website",
                    "retrieved_at": datetime.now().isoformat(),
                    "llm_provider": self.llm_provider
                }

            # Check cache first
            cached_result = await self.data_storage.get_cached_result(booking_id)
            if cached_result:
                print("✅ Found cached result!")
                logger.info(f"Found cached result for booking ID: {booking_id}")
                return cached_result

            # Perform AI tracking with the controlled browser session
            result = await self._perform_ai_tracking(booking_id)

            # Store successful interaction pattern
            if "error" not in result:
                await self.data_storage.store_interaction_pattern(booking_id, result)
                print("✅ Successfully retrieved shipping information!")
                logger.info(f"Successfully retrieved shipping information for booking ID: {booking_id}")
            
            return result

        except Exception as e:
            print(f"❌ Error tracking shipment: {str(e)}")
            logger.error(f"Error tracking shipment for booking ID {booking_id}: {str(e)}", exc_info=True)
            return {"error": str(e), "booking_id": booking_id}

    async def _perform_ai_tracking(self, booking_id: str) -> Dict:
        """
        Perform AI-driven tracking using the controlled browser session.
        """
        tracking_prompt = f"""
You are tasked with tracking a shipping container/booking with ID '{booking_id}'
on the seacargotracking.net website. Follow these specific steps in exact order:

Scroll in the steps of 100 pixels at a time.

STEP 1: INITIAL PAGE EXPLORATION
1. Navigate to http://www.seacargotracking.net
2. Examine the front page content but DO NOT use any tracking functionality on the main page
3. Ignore any tracking elements on the main page (search boxes, tracking buttons, container tracking sections)

STEP 2: LOCATE HMM (HYUNDAI MERCHANT MARINE)
4. Scroll down through the entire page to locate HMM (Hyundai Merchant Marine)
5. Look for shipping line logos, names, or links containing "HMM" or "Hyundai Merchant Marine"
6. Check all sections including footer, shipping line lists, or partner sections
7. Document the location where you found HMM as soon as you find HMM name just click on it do not scroll down

STEP 3: NAVIGATE TO HMM TRACKING PAGE  
8. Click on HMM (Hyundai Merchant Marine) or navigate to its section
9. Navigate to the HMM "Container Tracking" section
10. Wait for the HMM tracking page to fully load and examine all content

STEP 4: TRACKING EXECUTION
11. Look for tracking input forms on the HMM page (Track, trace, or other tracking forms)
12. Identify input field types (Container Number, Booking Number, BL Number)
13. Enter the booking ID: {booking_id} in the appropriate field
14. Click the "Track cargo" or "Search" button
15. Wait for results to load completely (at least 10 seconds)

STEP 5: RESULT VALIDATION AND TROUBLESHOOTING
16. If no results appear or "Not specified" appears:
    - Try the booking ID without any prefix (e.g., "SINI25432400" → "25432400")
    - Try different input fields if multiple are available
    - Look for error messages or "no results found" notifications

CRITICAL RULES:
- Never use tracking functionality from the front/main page
- Always locate and use HMM (Hyundai Merchant Marine) tracking system
- Only mark as successful if actual tracking data is displayed
- "Not specified" responses indicate tracking failure

EXPECTED TRACKING INFORMATION:
- Container/Booking status, Vessel name and voyage number
- Port of loading (POL) and Port of discharge (POD)  
- Departure and arrival dates, Current location
- Milestone events, Delivery status
- look at the whole page and find the information

RESPONSE FORMAT:
Return results in JSON format:
{{
 "tracking_status": "SUCCESS" or "FAILED",
 "booking_id": "{booking_id}",
 "exploration_log": "detailed log of pages visited and actions taken",
 "error_message": "if applicable",
 "container_details": {{
   "status": "actual status",
   "vessel_name": "actual vessel name", 
   "voyage_number": "actual voyage number",
   "port_of_loading": "actual port name",
   "port_of_discharge": "actual port name",
   "departure_date": "actual date",
   "arrival_date": "actual date",
   "current_location": "actual location",
   "milestones": ["list of actual events"],
   "delivery_status": "actual delivery status"
 }}
}}
"""

        logger.debug(f"Performing AI tracking for {booking_id} with controlled browser session")

        try:
            # Create agent with the controlled browser session
            tracking_agent = Agent(
                task=tracking_prompt,
                llm=self.llm,
                browser=await self.browser_controller.start_browser()
            )

            # Handle different LLM providers differently for agent execution
            if self.llm_provider == "gemini":
                # For Gemini, use synchronous execution wrapped in async
                logger.info(f"Using synchronous execution for Gemini LLM for booking ID: {booking_id}")
                result = await self._run_agent_sync(tracking_agent)
            else:
                # For other providers, use normal async execution
                logger.info(f"Using async execution for {self.llm_provider.upper()} LLM for booking ID: {booking_id}")
                result = await tracking_agent.run(max_steps=10)
            
            # Parse and structure the result
            structured_result = await self._parse_tracking_result(result, booking_id)
            return structured_result

        except Exception as e:
            logger.error(f"AI tracking failed for {booking_id}: {str(e)}", exc_info=True)
            return await self._fallback_tracking_method(booking_id)

    async def _run_agent_sync(self, agent):
        """
        Run agent synchronously to avoid async issues with Gemini.
        """
        def run_sync():
            try:
                # Create a new event loop for the synchronous execution
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Try to run the agent synchronously
                    if hasattr(agent, 'run_sync'):
                        return agent.run_sync(max_steps=10)
                    else:
                        # Fallback: run async in the new loop
                        return loop.run_until_complete(agent.run(max_steps=10))
                finally:
                    loop.close()
            except Exception as e:
                logger.error(f"Sync agent execution failed: {str(e)}")
                raise
        
        # Run in thread pool to avoid blocking
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_sync)
            return await asyncio.wrap_future(future)

    async def _parse_tracking_result(self, agent_history, booking_id: str) -> Dict:
        """
        Parse the result from AI browser automation into structured data.

        Args:
            agent_history: AgentHistoryList object from browser automation
            booking_id (str): Original booking ID

        Returns:
            Dict: Structured shipping information
        """
        # Extract text content from agent history
        try:
            logger.debug(f"Extracting text from agent history for {booking_id}")
            if hasattr(agent_history, 'final_result'):
                raw_result = str(agent_history.final_result)
            elif hasattr(agent_history, 'content'):
                raw_result = str(agent_history.content)
            elif hasattr(agent_history, '__iter__'):
                # Combine all action results
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
        - current_location: Current location of the shipment
        - milestones: List of tracking events/milestones
        - delivery_status: Delivery status information

        Return as valid JSON only, no additional text.
        """

        logger.debug(f"Parsing raw result with {self.llm_provider.upper()} LLM for {booking_id}")

        try:
            logger.info(f"Invoking {self.llm_provider.upper()} LLM for data extraction for booking ID: {booking_id}")
            
            # Handle different LLM providers for parsing
            if self.llm_provider == "gemini":
                # Use synchronous invoke for Gemini
                response = await self._invoke_llm_sync(extraction_prompt)
            else:
                # Use async invoke for other providers
                response = await self.llm.ainvoke(extraction_prompt)
                
            extracted_data = response.content.strip()
            logger.debug(f"{self.llm_provider.upper()} LLM raw extracted_data for {booking_id}: {extracted_data[:200]}...")

            # Clean the response to ensure it's valid JSON
            if extracted_data.startswith('```'):
                extracted_data = extracted_data.replace('```', '').replace('json', '')

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
            logger.error(f"Failed to parse {self.llm_provider.upper()} LLM result for {booking_id}: {str(e)}", exc_info=True)
            return {
                "booking_id": booking_id,
                "error": f"Failed to parse result: {str(e)}",
                "raw_result": str(raw_result),
                "retrieved_at": datetime.now().isoformat(),
                "llm_provider": self.llm_provider
            }

    async def _invoke_llm_sync(self, prompt):
        """
        Invoke LLM synchronously for Gemini to avoid async issues.
        """
        def invoke_sync():
            try:
                return self.llm.invoke(prompt)
            except Exception as e:
                logger.error(f"Sync LLM invocation failed: {str(e)}")
                raise
        
        # Run in thread pool to avoid blocking
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(invoke_sync)
            return await asyncio.wrap_future(future)

    async def _fallback_tracking_method(self, booking_id: str) -> Dict:
        """
        Fallback method when primary AI tracking fails.

        Args:
            booking_id (str): The booking ID to track

        Returns:
            Dict: Tracking result or error information
        """
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
        
        fallback_agent = Agent(
            task=fallback_prompt,
            llm=self.llm,
            browser=await self.browser_controller.start_browser()
        )

        try:
            logger.info(f"Running fallback BrowserAgent for {booking_id} with max_steps=8")
            
            # Handle different LLM providers for fallback
            if self.llm_provider == "gemini":
                result = await self._run_agent_sync(fallback_agent)
            else:
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
        """
        Track multiple shipments concurrently.

        Args:
            booking_ids (list): List of booking IDs to track

        Returns:
            Dict[str, Dict]: Results for each booking ID
        """
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

    async def __aenter__(self):
        """Async context manager entry."""
        await self._setup_browser_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._cleanup_browser_session()

async def main():
    """
    Main function to run the shipping tracking agent from command line.
    """
    if len(sys.argv) != 2:
        print("Usage: python agent.py <booking_id>")
        print("Example: python agent.py SINI25432400")
        sys.exit(1)
    
    booking_id = sys.argv[1]
    
    try:
        # Use as context manager for automatic cleanup
        async with ShippingTrackingAgent(headless=False) as agent:
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
        
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        logger.error(f"Fatal error in main: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
