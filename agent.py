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

# Local imports
from browser_controller import BrowserController
from data_storage import DataStorage
from config import Config

# Load environment variables
load_dotenv()

# Global configuration for LLM provider
# Set this to 'openai' or 'gemini' to switch between providers
LLM_PROVIDER = "groq"#os.getenv("LLM_PROVIDER", "openai").lower()  # Default to OpenAI

# Set up basic logging configuration
# This will be configured more robustly in main if this script is run directly
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ShippingTrackingAgent:
    """
    Main AI agent for shipping line tracking.

    This agent uses natural language processing to interact with websites
    and extract shipping information without hardcoded interactions.
    """

    def __init__(self, llm_provider: str = None):
        """
        Initialize the shipping tracking agent.
        
        Args:
            llm_provider (str, optional): Override the global LLM provider ('openai' or 'gemini')
        """
        logger.info("Initializing ShippingTrackingAgent...")
        self.config = Config()
        self.data_storage = DataStorage()
        self.browser_controller = BrowserController()

        # Use provided LLM provider or fall back to global setting
        self.llm_provider = (llm_provider or LLM_PROVIDER).lower()
        
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
            
            # Check for required environment variable
            if not os.getenv("GOOGLE_API_KEY"):
                raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini")
            
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-pro-preview-03-25",
                temperature=0.5,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                convert_system_message_to_human=True  # Gemini compatibility
            )
            
        elif self.llm_provider == "openai":
            logger.info("Initializing OpenAI LLM...")
            
            # Check for required environment variable
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI")
            
            return ChatOpenAI(
                model="gpt-4o",
                temperature=0.1,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        
        elif self.llm_provider == "groq":
            logger.info("Initializing Groq LLM...")
            
            # Check for required environment variable
            if not os.getenv("GROQ_API_KEY"):
                raise ValueError("GROQ_API_KEY environment variable is required for Groq")
            
            return ChatGroq(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0.1,
                api_key=os.getenv("GROQ_API_KEY")
            )
            
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}. Supported providers: 'openai', 'gemini'")

    def get_llm_info(self) -> Dict[str, str]:
        """
        Get information about the current LLM configuration.
        
        Returns:
            Dict containing LLM provider and model information
        """
        if self.llm_provider == "gemini":
            return {
                "provider": "Google Gemini",
                "model": "gemini-2.5-pro-preview-03-25",
                "api_key_set": bool(os.getenv("GOOGLE_API_KEY"))
            }
        elif self.llm_provider == "openai":
            return {
                "provider": "OpenAI",
                "model": "gpt-4o", 
                "api_key_set": bool(os.getenv("OPENAI_API_KEY"))
            }
        elif self.llm_provider == "groq":
            return {
                "provider": "Groq",
                "model": "llama-4-scout-17b-16e-instruct",
                "api_key_set": bool(os.getenv("GROQ_API_KEY"))
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
on the seacargotracking.net website. Follow these specific steps:

STEP 1: INITIAL PAGE EXPLORATION
1. Navigate to http://www.seacargotracking.net
2. **CRITICAL**: First, thoroughly examine the entire front page content
3. Look for ANY tracking-related elements on the main page:
   - Search boxes or input fields
   - Tracking buttons or links
   - Container tracking sections
   - Dropdown menus for shipping lines
   - Any direct tracking functionality

STEP 2: FRONT PAGE TRACKING ATTEMPT
4. If you find ANY tracking functionality on the front page:
   - Try using it first with the booking ID: {booking_id}
   - Test different input field options if multiple are available
   - Document what you find and whether it works

STEP 3: NAVIGATION TO SPECIFIC SECTIONS
5. If front page tracking doesn't work or isn't available, then proceed with navigation:
   - Click on "Container Tracking" (NOT "Track & Trace")
   - Explore the page that opens and check its full content
   
STEP 4: SHIPPING LINE SELECTION
6. Look for shipping line selection options:
   - Scroll down to find the shipping line selection
   - Click on "HYUNDAI Merchant Marine (HMM)" link
   - Wait for the page to fully load and examine all content

STEP 5: TRACKING FORM INTERACTION
7. After clicking HMM, thoroughly check the new page:
   - Look for ALL tracking input forms that appear
   - Identify different input field types (Container Number, Booking Number, BL Number)
   - Try the booking ID: {booking_id} in the most appropriate field first

STEP 6: TRACKING EXECUTION
8. In the tracking form:
   - Enter the booking ID: {booking_id}
   - Make sure to use the correct input field
   - Click the "Track cargo" or "Search" button
   - Wait for results to load completely (at least 10 seconds)

STEP 7: RESULT VALIDATION AND TROUBLESHOOTING
9. If no results appear or "Not specified" appears:
   - Try using the booking ID without any prefix (e.g., if booking is "SINI25432400", try "25432400")
   - Try different input fields if multiple are available
   - Look for error messages or "no results found" notifications
   - Go back and try other tracking options found on previous pages

CRITICAL VALIDATION STEPS:
- Before marking as complete, verify that actual tracking data is displayed
- Look for specific vessel names, dates, port names, or status information
- If you see "Not specified" for all fields, the tracking has FAILED
- Document each step you took and what you found on each page

FALLBACK OPTION:
- If tracking fails on seacargotracking.net, try the HMM official website: 
  https://www.hmm21.com/e-service/general/trackNTrace/TrackNTrace.do

EXPECTED TRACKING INFORMATION TO EXTRACT:
- Container/Booking status (In Transit, Discharged, Delivered, etc.)
- Vessel name and voyage number
- Port of loading (POL) and Port of discharge (POD)
- Actual/Estimated departure and arrival dates
- Current location of container
- Milestone events (Gate In, Loaded, Discharged, etc.)
- Delivery status and final destination

ERROR HANDLING:
- If booking ID is not found, clearly state "Booking ID not found in system"
- If website is down or not responding, try alternative tracking methods
- If tracking partially works but some data is missing, specify which fields are available vs missing
- Include any error messages from the website in your response
- Document your exploration process for each page visited

RESPONSE FORMAT:
Return results in JSON format with actual values, not "Not specified":
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

IMPORTANT: 
- Thoroughly explore each page before moving to the next step
- Only mark tracking as successful if you receive actual, specific tracking information
- Generic "Not specified" responses indicate tracking failure
- Document your exploration process to help with troubleshooting
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

# CLI Interface
async def main():
    """Main function for command-line interface."""

    # Configure logging more robustly
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_file = os.getenv("LOG_FILE", "shipping_tracking.log")

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_str)

    # Remove any existing handlers to avoid duplicate logs if re-run in same session
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level_str) # Respect LOG_LEVEL for console too
    root_logger.addHandler(console_handler)

    # File Handler
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a') # Append mode
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(log_level_str) # Respect LOG_LEVEL for file
        root_logger.addHandler(file_handler)
    
    logger.info("Logging configured for CLI execution.")

    if len(sys.argv) < 2:
        print("Usage: python agent.py <booking_id> [booking_id2] [booking_id3] ...")
        print("Example: python agent.py SINI25432400")
        print("Example: python agent.py SINI25432400 ABCD12345678")
        print(f"\nCurrent LLM Provider: {LLM_PROVIDER.upper()}")
        print("Set LLM_PROVIDER environment variable to 'openai' or 'gemini' to change provider")
        sys.exit(1)

    booking_ids = sys.argv[1:]

    try:
        # Initialize the agent
        agent = ShippingTrackingAgent()
        
        # Display LLM configuration
        llm_info = agent.get_llm_info()
        print(f"\n🤖 LLM Configuration:")
        print(f"   Provider: {llm_info['provider']}")
        print(f"   Model: {llm_info['model']}")
        print(f"   API Key Set: {'✅' if llm_info['api_key_set'] else '❌'}")
        
        if not llm_info['api_key_set']:
            print(f"❌ Error: API key not set for {llm_info['provider']}")
            print(f"Please set the appropriate environment variable:")
            if agent.llm_provider == "gemini":
                print("   export GOOGLE_API_KEY=your_gemini_api_key")
            else:
                print("   export OPENAI_API_KEY=your_openai_api_key")
            sys.exit(1)

    except ValueError as e:
        print(f"❌ Configuration Error: {str(e)}")
        print(f"\nAvailable LLM providers: 'openai', 'gemini'")
        print(f"Current setting: LLM_PROVIDER={LLM_PROVIDER}")
        print(f"To change provider, set environment variable: export LLM_PROVIDER=gemini")
        sys.exit(1)

    logger.info(f"Tracking {'single' if len(booking_ids) == 1 else 'multiple'} shipment(s): {', '.join(booking_ids)}")

    if len(booking_ids) == 1:
        # Single shipment tracking
        result = await agent.track_shipment(booking_ids[0])
        
        # Display results
        print("\n" + "="*50)
        print("TRACKING RESULTS")
        print("="*50)

        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"📦 Booking ID: {result.get('booking_id', 'N/A')}")
            print(f"🚢 Voyage Number: {result.get('voyage_number', 'N/A')}")
            print(f"📅 Arrival Date: {result.get('arrival_date', 'N/A')}")
            print(f"🏭 Vessel Name: {result.get('vessel_name', 'N/A')}")
            print(f"🔄 Status: {result.get('status', 'N/A')}")
            if result.get('port_of_loading'):
                print(f"🚢 Port of Loading: {result.get('port_of_loading', 'N/A')}")
            if result.get('port_of_discharge'):
                print(f"🏭 Port of Discharge: {result.get('port_of_discharge', 'N/A')}")
            print(f"🤖 LLM Provider: {result.get('llm_provider', 'N/A').upper()}")
    else:
        # Multiple shipments tracking
        results = await agent.track_multiple_shipments(booking_ids)
        
        logger.info("Batch tracking results processed.")
        print("\n" + "="*50)
        print("BATCH TRACKING RESULTS")
        print("="*50)
        
        for booking_id, result in results.items():
            print(f"\n📦 Booking ID: {booking_id}")
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"🚢 Voyage Number: {result.get('voyage_number', 'N/A')}")
                print(f"📅 Arrival Date: {result.get('arrival_date', 'N/A')}")
                print(f"🔄 Status: {result.get('status', 'N/A')}")
                print(f"🤖 LLM Provider: {result.get('llm_provider', 'N/A').upper()}")

    print("\n✅ Tracking completed!")
    logger.info("Tracking completed!")

if __name__ == "__main__":
    asyncio.run(main())