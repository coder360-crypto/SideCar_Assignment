import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, Optional, Tuple
import logging

# Third-party imports
from browser_use import Agent
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Local imports - make these optional to avoid Config issues
try:
    from browser_controller import BrowserController
except ImportError:
    BrowserController = None
    
try:
    from data_storage import DataStorage
except ImportError:
    DataStorage = None

# Load environment variables
load_dotenv()

# Global configuration for LLM provider
# Set this to 'openai' or 'gemini' to switch between providers
LLM_PROVIDER = "gemini"#os.getenv("LLM_PROVIDER", "openai").lower()  # Default to OpenAI

# Set up basic logging configuration
# This will be configured more robustly in main if this script is run directly
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleDataStorage:
    """Simple data storage class to replace the problematic DataStorage import."""
    
    def __init__(self):
        self.cache = {}
        
    async def get_cached_result(self, booking_id: str):
        """Simple cache lookup - always returns None for now."""
        return None
        
    async def store_interaction_pattern(self, booking_id: str, result: dict):
        """Simple cache storage."""
        self.cache[booking_id] = result
        logger.info(f"Stored interaction pattern for {booking_id}")

class SimpleBrowserController:
    """Simple browser controller to replace the problematic BrowserController import."""
    
    def __init__(self):
        pass

class ShippingTrackingAgent:
    """
    Main AI agent for shipping line tracking.

    This agent uses natural language processing to interact with websites
    and extract shipping information without hardcoded interactions.
    """

    def __init__(self, llm_provider: str = None, api_key: str = None):
        """
        Initialize the shipping tracking agent.
        
        Args:
            llm_provider (str, optional): Override the global LLM provider ('openai' or 'gemini')
            api_key (str, optional): API key for the chosen LLM provider
        """
        logger.info("Initializing ShippingTrackingAgent...")
        
        # Use simple replacements instead of the problematic imports
        self.data_storage = SimpleDataStorage() if DataStorage is None else DataStorage()
        self.browser_controller = SimpleBrowserController() if BrowserController is None else BrowserController()

        # Use provided LLM provider or fall back to global setting
        self.llm_provider = (llm_provider or LLM_PROVIDER).lower()
        
        # Store the API key
        self.api_key = api_key
        
        # Initialize the appropriate LLM based on provider
        self.llm = self._initialize_llm()
        
        logger.info(f"ShippingTrackingAgent initialized with {self.llm_provider.upper()} LLM.")

    def _initialize_llm(self):
        """
        Initialize the appropriate LLM based on the provider setting.
        
        Returns:
            LLM instance (ChatOpenAI or ChatGoogleGenerativeAI)
        """
        if self.llm_provider == "gemini":
            logger.info("Initializing Gemini LLM...")
            
            # Use provided API key or environment variable
            api_key = self.api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable or api_key parameter is required for Gemini")
            
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-preview-05-20",  # Updated to use the latest available model
                temperature=0.5,
                google_api_key=api_key,
               # convert_system_message_to_human=True  # Gemini compatibility
            )
            
        elif self.llm_provider == "openai":
            logger.info("Initializing OpenAI LLM...")
            
            # Use provided API key or environment variable
            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable or api_key parameter is required for OpenAI")
            
            return ChatOpenAI(
                model="o4-mini-2025-04-16",
               # temperature=0.1,
                api_key=api_key
            )
        
        elif self.llm_provider == "groq":
            logger.info("Initializing Groq LLM...")
            
            # Use provided API key or environment variable
            api_key = self.api_key or os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable or api_key parameter is required for Groq")
            
            return ChatGroq(
                model="meta-llama/llama-4-scout-17b-16e-instruct",  # Updated to use a more widely available model
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
                "api_key_set": bool(self.api_key or "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")
            }

    async def track_shipment(self, booking_id: str) -> Dict:
        """
        Track a shipment using the provided booking ID.

        Args:
            booking_id (str): The booking ID to track (e.g., SINI25432400)

        Returns:
            Dict: Contains voyage_number, arrival_date, and metadata
        """
        print(f"\n🚢 Starting shipment tracking for booking ID: {booking_id}")
        print(f"🤖 Using {self.get_llm_info()['provider']} ({self.get_llm_info()['model']})")
        logger.info(f"Starting shipment tracking for booking ID: {booking_id} with {self.llm_provider}")

        try:
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

    async def _perform_ai_tracking(self, booking_id: str) -> Dict:
        """
        Perform AI-driven tracking using natural language instructions.

        Args:
            booking_id (str): The booking ID to track

        Returns:
            Dict: Extracted shipping information
        """

        # Natural language prompt for the AI agent
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

        logger.debug(f"Performing AI tracking for {booking_id}. Prompt: {tracking_prompt[:200]}...")

        # Create a new agent with the specific task
        logger.info(f"Creating BrowserAgent for task: {tracking_prompt[:100]}...")
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
            # Get the final result or combine all steps
            logger.debug(f"Extracting text from agent history for {booking_id}")
            if hasattr(agent_history, 'final_result'):
                raw_result = str(agent_history.final_result)
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

        Return as valid JSON only, no additional text.
        """

        logger.debug(f"Parsing raw result with {self.llm_provider.upper()} LLM for {booking_id}. Extraction prompt: {extraction_prompt[:200]}...")

        try:
            # Use invoke instead of agenerate for better compatibility
            logger.info(f"Invoking {self.llm_provider.upper()} LLM for data extraction for booking ID: {booking_id}")
            response = self.llm.invoke(extraction_prompt)
            extracted_data = response.content.strip()
            logger.debug(f"{self.llm_provider.upper()} LLM raw extracted_data for {booking_id}: {extracted_data[:200]}...")

            # Clean the response to ensure it's valid JSON
            if extracted_data.startswith('```json'):
                extracted_data = extracted_data.replace('```json', '').replace('```', '').strip()

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
            logger.error(f"Failed to parse {self.llm_provider.upper()} LLM result for {booking_id}: {str(e)}. Raw extracted data: {extracted_data[:500] if 'extracted_data' in locals() else 'N/A'}", exc_info=True)
            return {
                "booking_id": booking_id,
                "error": f"Failed to parse result: {str(e)}",
                "raw_result": str(raw_result),
                "retrieved_at": datetime.now().isoformat(),
                "llm_provider": self.llm_provider
            }

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

        # Create a new agent instance for fallback to ensure clean state
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
            llm=self.llm
        )

        try:
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