import asyncio
import os
import sys
import signal
import atexit
from datetime import datetime
from typing import Dict, Optional, Tuple
import logging

# Third-party imports
from browser_use import Agent
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from browser_controller import BrowserController

from data_storage import DataStorage

# Load environment variables
load_dotenv()

# Global configuration for LLM provider
LLM_PROVIDER = "gemini"

# Set up basic logging configuration
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variable to track active agents for cleanup
_active_agents = []
_cleanup_registered = False

def register_cleanup_handlers():
    """Register cleanup handlers for proper shutdown."""
    global _cleanup_registered
    if _cleanup_registered:
        return
    
    def cleanup_on_exit():
        """Cleanup function for atexit."""
        if _active_agents:
            logger.info("Exit cleanup: Cleaning up remaining agents...")
            # Run cleanup in a new event loop since we might not have one
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(_cleanup_all_agents())
                loop.close()
            except Exception as e:
                logger.error(f"Error during exit cleanup: {e}")
    
    def signal_handler(signum, frame):
        """Handle interrupt signals."""
        logger.info(f"Received signal {signum}, cleaning up...")
        cleanup_on_exit()
        sys.exit(0)
    
    # Register cleanup handlers
    atexit.register(cleanup_on_exit)
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination
    
    _cleanup_registered = True
    logger.info("Cleanup handlers registered")

async def _cleanup_all_agents():
    """Clean up all active agents globally."""
    global _active_agents
    logger.info(f"Cleaning up {len(_active_agents)} active agents...")
    
    cleanup_tasks = []
    for agent in _active_agents:
        cleanup_tasks.append(_cleanup_single_agent(agent))
    
    if cleanup_tasks:
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
    
    _active_agents.clear()
    logger.info("All agents cleaned up")

async def _cleanup_single_agent(agent):
    """Clean up a single agent with better error handling."""
    try:
        # Force close browser if it exists
        if hasattr(agent, 'browser') and agent.browser:
            try:
                await asyncio.wait_for(agent.browser.close(), timeout=5.0)
                logger.debug("Browser closed for agent")
            except asyncio.TimeoutError:
                logger.warning("Browser close timeout for agent")
            except Exception as e:
                logger.warning(f"Error closing browser for agent: {e}")
            
        # Close browser context if it exists
        if hasattr(agent, 'browser_context') and agent.browser_context:
            try:
                await asyncio.wait_for(agent.browser_context.close(), timeout=3.0)
                logger.debug("Browser context closed for agent")
            except asyncio.TimeoutError:
                logger.warning("Browser context close timeout for agent")
            except Exception as e:
                logger.warning(f"Error closing browser context for agent: {e}")
            
        # Stop playwright if it exists
        if hasattr(agent, 'playwright') and agent.playwright:
            try:
                await asyncio.wait_for(agent.playwright.stop(), timeout=3.0)
                logger.debug("Playwright stopped for agent")
            except asyncio.TimeoutError:
                logger.warning("Playwright stop timeout for agent")
            except Exception as e:
                logger.warning(f"Error stopping playwright for agent: {e}")
                
    except Exception as e:
        logger.warning(f"Error cleaning up agent: {e}")


class ShippingTrackingAgent:
    """
    Main AI agent for shipping line tracking with improved cleanup.
    """

    def __init__(self, llm_provider: str = None, api_key: str = None):
        """Initialize the shipping tracking agent with cleanup handlers."""
        logger.info("Initializing ShippingTrackingAgent...")
        
        # Register cleanup handlers on first initialization
        register_cleanup_handlers()
        
        # Use simple replacements instead of the problematic imports
        self.data_storage =  DataStorage()
        self.browser_controller =  BrowserController()

        # Use provided LLM provider or fall back to global setting
        self.llm_provider = (llm_provider or LLM_PROVIDER).lower()
        
        # Store the API key
        self.api_key = api_key 
        
        # Initialize the appropriate LLM based on provider
        self.llm = self._initialize_llm()
        
        # Keep track of active agents for cleanup
        self.active_agents = []
        
        logger.info(f"ShippingTrackingAgent initialized with {self.llm_provider.upper()} LLM.")

    def _initialize_llm(self):
        """Initialize the appropriate LLM based on the provider setting."""
        if self.llm_provider == "gemini":
            logger.info("Initializing Gemini LLM...")
            
            api_key = self.api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable or api_key parameter is required for Gemini")
            
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-preview-05-20",
                temperature=0.5,
                google_api_key=api_key
            )
            
        elif self.llm_provider == "openai":
            logger.info("Initializing OpenAI LLM...")
            
            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable or api_key parameter is required for OpenAI")
            
            return ChatOpenAI(
                model="gpt-4o-mini",  # Fixed model name
                temperature=0.1,
                api_key=api_key
            )
        
        elif self.llm_provider == "groq":
            logger.info("Initializing Groq LLM...")
            
            api_key = self.api_key or os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable or api_key parameter is required for Groq")
            
            return ChatGroq(
                model="llama-3.1-70b-versatile",  # Fixed model name
                temperature=0.2,
                api_key=api_key
            )
            
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}. Supported providers: 'openai', 'gemini', 'groq'")

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
                "model": "gpt-4o-mini", 
                "api_key_set": bool(self.api_key or os.getenv("OPENAI_API_KEY"))
            }
        elif self.llm_provider == "groq":
            return {
                "provider": "Groq",
                "model": "llama-3.1-70b-versatile",
                "api_key_set": bool(self.api_key or os.getenv("GROQ_API_KEY"))
            }

    async def _cleanup_agents(self):
        """Clean up all active browser agents with timeout."""
        global _active_agents
        
        logger.info(f"Cleaning up {len(self.active_agents)} active agents...")
        
        # Create cleanup tasks with timeout
        cleanup_tasks = []
        for agent in self.active_agents:
            cleanup_tasks.append(_cleanup_single_agent(agent))
        
        if cleanup_tasks:
            try:
                # Wait for cleanup with timeout
                await asyncio.wait_for(
                    asyncio.gather(*cleanup_tasks, return_exceptions=True),
                    timeout=15.0  # Increased timeout for cleanup
                )
            except asyncio.TimeoutError:
                logger.warning("Cleanup timeout reached, some resources might not be cleaned up properly")
        
        # Remove from global tracking
        for agent in self.active_agents:
            if agent in _active_agents:
                _active_agents.remove(agent)
        
        self.active_agents.clear()
        logger.info("All agents cleaned up")

    async def track_shipment(self, booking_id: str) -> Dict:
        """Track a shipment using the provided booking ID with improved cleanup."""
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
            
            # Explicitly close browser after successful tracking
            await self._explicit_browser_close()
            
            return result

        except Exception as e:
            print(f"❌ Error tracking shipment: {str(e)}")
            logger.error(f"Error tracking shipment for booking ID {booking_id}: {str(e)}", exc_info=True)
            return {"error": str(e), "booking_id": booking_id}
        finally:
            # Always clean up agents after tracking
            await self._cleanup_agents()


    async def _perform_ai_tracking(self, booking_id: str) -> Dict:
        """Perform AI-driven tracking with better resource management and auto-close."""
        global _active_agents

        # Enhanced natural language prompt with explicit completion instructions
        tracking_prompt = f"""
You are tasked with tracking a shipping container/booking with ID '{booking_id}'
on the seacargotracking.net website. Follow these specific steps in exact order:

STEP 1: INITIAL PAGE EXPLORATION
1. Navigate to http://www.seacargotracking.net
2. Examine the front page content but DO NOT use any tracking functionality on the main page
3. Ignore any tracking elements on the main page (search boxes, tracking buttons, container tracking sections)

STEP 2: LOCATE HMM (HYUNDAI MERCHANT MARINE)
4. Scroll down through the entire page to locate HMM (Hyundai Merchant Marine)
5. Look for shipping line logos, names, or links containing "HMM" or "Hyundai Merchant Marine"
6. Check all sections including footer, shipping line lists, or partner sections
7. Once you find HMM, click on it immediately

STEP 3: NAVIGATE TO HMM TRACKING PAGE  
8. Click on HMM (Hyundai Merchant Marine) or navigate to its section
9. Navigate to the HMM "Container Tracking" section
10. Wait for the HMM tracking page to fully load and examine all content

STEP 4: TRACKING EXECUTION
11. Look for tracking input forms on the HMM page
12. Identify input field types (Container Number, Booking Number, BL Number)
13. Enter the booking ID: {booking_id} in the appropriate field
14. Click the "Track cargo" or "Search" button
15. Wait for results to load completely (at least 10 seconds)

STEP 5: RESULT EXTRACTION AND COMPLETION
16. Extract all visible tracking information from the results page
17. Look for: vessel name, voyage number, departure date, arrival date, port information, status
18. If no results appear or "Not specified" appears, try different input fields
19. Once you have extracted all available information, IMMEDIATELY CLOSE THE BROWSER
20. DO NOT wait for user input - close the browser automatically after extracting data

CRITICAL COMPLETION RULES:
- ALWAYS close the browser after completing the tracking task
- DO NOT leave the browser open waiting for user input
- Extract data and close immediately
- Never use tracking functionality from the main page - always use HMM tracking
- Return JSON format with all extracted information

BROWSER CLOSING INSTRUCTION:
After extracting all tracking information, immediately execute browser close commands.
Do not wait for any user confirmation or input.

You have to use the browser controller to close the browser.

Its the most important thing to do.
"""

        logger.debug(f"Performing AI tracking for {booking_id}")

        # Create a new agent with timeout and cleanup
        tracking_agent = None
        try:
            logger.info(f"Creating BrowserAgent for booking ID: {booking_id}")
            
            # Create agent with explicit headless mode and close_browser_on_finish
            tracking_agent = Agent(
                task=tracking_prompt,
                llm=self.llm,

            )
            
            # Add to both local and global tracking for cleanup
            self.active_agents.append(tracking_agent)
            _active_agents.append(tracking_agent)

            # Execute with timeout
            logger.info(f"Running BrowserAgent for booking ID: {booking_id}")
            result = await asyncio.wait_for(
                tracking_agent.run(max_steps=15),  # Increased steps for completion
                timeout=180.0  # 3 minute timeout
            )
            
            logger.debug(f"BrowserAgent completed for {booking_id}")

            # Parse and structure the result
            structured_result = await self._parse_tracking_result(result, booking_id)
            logger.info(f"Successfully performed AI tracking for {booking_id}")
            return structured_result

        except asyncio.TimeoutError:
            logger.error(f"AI tracking timeout for {booking_id}")
            return {
                "booking_id": booking_id,
                "error": "Tracking timeout - operation took too long",
                "retrieved_at": datetime.now().isoformat(),
                "llm_provider": self.llm_provider
            }
        except Exception as e:
            print(f"Primary tracking failed: {str(e)}")
            logger.error(f"Primary AI tracking failed for {booking_id}: {str(e)}", exc_info=True)
            return {
                "booking_id": booking_id,
                "error": f"AI tracking failed: {str(e)}",
                "retrieved_at": datetime.now().isoformat(),
                "llm_provider": self.llm_provider
            }
        finally:
            # Immediate cleanup of this specific agent
            if tracking_agent:
                await _cleanup_single_agent(tracking_agent)
                if tracking_agent in self.active_agents:
                    self.active_agents.remove(tracking_agent)
                if tracking_agent in _active_agents:
                    _active_agents.remove(tracking_agent)
                logger.info(f"Agent cleanup completed for {booking_id}")
                
                
                
    async def _explicit_browser_close(self):
        """Explicitly close all browser instances and resources."""
        logger.info("Explicitly closing browser instances...")
        
        try:
            # Close browser controller if it exists
            if hasattr(self.browser_controller, 'close_browser'):
                await self.browser_controller.close_browser()
                logger.info("BrowserController explicitly closed")
            
            # Force close any remaining browser instances
            for agent in self.active_agents:
                if hasattr(agent, 'browser') and agent.browser:
                    try:
                        await agent.browser.close()
                        logger.info("Browser instance explicitly closed")
                    except Exception as e:
                        logger.warning(f"Error explicitly closing browser: {e}")
            
            print("🔒 Browser instances explicitly closed")
            
        except Exception as e:
            logger.error(f"Error in explicit browser close: {e}")


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
                
        except Exception as e:
            logger.warning(f"Could not extract text from agent_history: {e}")
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

        try:
            logger.info(f"Invoking LLM for data extraction for booking ID: {booking_id}")
            response = self.llm.invoke(extraction_prompt)
            extracted_data = response.content.strip()

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

            return parsed_data

        except Exception as e:
            logger.error(f"Failed to parse LLM result for {booking_id}: {str(e)}")
            return {
                "booking_id": booking_id,
                "error": f"Failed to parse result: {str(e)}",
                "raw_result": str(raw_result),
                "retrieved_at": datetime.now().isoformat(),
                "llm_provider": self.llm_provider
            }

    async def track_multiple_shipments(self, booking_ids: list) -> Dict[str, Dict]:
        """Track multiple shipments sequentially."""
        print(f"\n🚢 Starting batch tracking for {len(booking_ids)} shipments...")
        logger.info(f"Starting batch tracking for {len(booking_ids)} shipments")
        
        results = {}
        
        for booking_id in booking_ids:
            try:
                logger.info(f"Processing booking ID: {booking_id}")
                result = await self.track_shipment(booking_id)
                results[booking_id] = result
                
                # Add delay between requests
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error processing booking ID {booking_id}: {str(e)}")
                results[booking_id] = {
                    "error": str(e), 
                    "booking_id": booking_id, 
                    "llm_provider": self.llm_provider
                }
        
        return results

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with forced cleanup."""
        logger.info("ShippingTrackingAgent context manager cleanup...")
        await self._cleanup_agents()
        
        # Also clean up browser controller if it exists
        if hasattr(self.browser_controller, 'close_browser'):
            try:
                await self.browser_controller.close_browser()
                logger.info("BrowserController cleaned up")
            except Exception as e:
                logger.warning(f"Error cleaning up BrowserController: {e}")

async def main():
    """Main function with improved error handling and cleanup."""
    if len(sys.argv) != 2:
        print("Usage: python agent.py <booking_id>")
        print("Example: python agent.py SINI25432400")
        sys.exit(1)
    
    booking_id = sys.argv[1]
    
    # Use the agent as a context manager with timeout
    try:
        async with ShippingTrackingAgent() as agent:
            # Track the shipment with overall timeout
            result = await asyncio.wait_for(
                agent.track_shipment(booking_id),
                timeout=300.0  # 5 minute overall timeout
            )
            
            # Print the results
            print("\n" + "="*60)
            print("📦 SHIPPING TRACKING RESULTS")
            print("="*60)
            
            if "error" in result:
                print(f"❌ Tracking failed for booking ID: {booking_id}")
                print(f"Error: {result['error']}")
            else:
                print(f"✅ Tracking successful for booking ID: {booking_id}")
                print(f"🤖 LLM Provider: {result.get('llm_provider', 'Unknown')}")
                print(f"📅 Retrieved at: {result.get('retrieved_at', 'Unknown')}")
                
                # Print tracking information
                print("📋 TRACKING INFORMATION:")
                print(f"   Voyage Number: {result.get('voyage_number', 'N/A')}")
                print(f"   Arrival Date: {result.get('arrival_date', 'N/A')}")
                print(f"   Departure Date: {result.get('departure_date', 'N/A')}")
                print(f"   Vessel Name: {result.get('vessel_name', 'N/A')}")
                print(f"   Port of Loading: {result.get('port_of_loading', 'N/A')}")
                print(f"   Port of Discharge: {result.get('port_of_discharge', 'N/A')}")
                print(f"   Status: {result.get('status', 'N/A')}")
            
            print("="*60)
            print("🔄 Cleaning up browser resources...")
            
    except asyncio.TimeoutError:
        print("❌ Overall operation timeout - cleaning up...")
        logger.error("Main operation timeout")
        await _cleanup_all_agents()
    except KeyboardInterrupt:
        print("❌ Operation interrupted - cleaning up...")
        logger.info("Operation interrupted by user")
        await _cleanup_all_agents()
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        logger.error(f"Fatal error in main: {str(e)}", exc_info=True)
        await _cleanup_all_agents()
        sys.exit(1)
    finally:
        # Final cleanup to ensure everything is closed
        await _cleanup_all_agents()
        print("✅ All resources cleaned up. Program completed.")

if __name__ == "__main__":
    # Ensure proper cleanup on exit
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        sys.exit(1)