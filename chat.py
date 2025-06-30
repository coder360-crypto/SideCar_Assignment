import asyncio
import re
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable
import logging
import chainlit as cl
from io import StringIO
import sys
import threading
from queue import Queue, Empty

# Import your existing agent
from agent import ShippingTrackingAgent

logger = logging.getLogger(__name__)

class TerminalLogCapture:
    """Captures ALL terminal logs from the agent and its dependencies."""
    
    # Add a constant for max log length
    MAX_LOG_LENGTH = 200  # Truncate logs to this length

    class _QueueLogHandler(logging.Handler):
        def __init__(self, queue: Queue, max_length: int = 200):
            super().__init__()
            self.queue = queue
            self.max_length = max_length

        def emit(self, record: logging.LogRecord):
            try:
                msg = self.format(record)
                # Skip log if it's too long
                if len(msg) > self.max_length:
                    return
                    
                self.queue.put({
                    'level': record.levelname,
                    'message': msg,
                    'timestamp': datetime.fromtimestamp(record.created),
                    'logger_name': record.name
                })
            except Exception:
                pass

    def __init__(self):
        self.log_queue = Queue()
        self.active_handlers = []
        self.original_levels = {}
        self.capturing = False
        
    def start_capture(self):
        """Start capturing ALL logs from the root logger and all child loggers."""
        if self.capturing:
            return
            
        self.capturing = True
        
        # Get the root logger to capture everything
        root_logger = logging.getLogger()
        
        # Store original level
        self.original_levels['root'] = root_logger.level
        
        # Set root logger to capture all levels
        root_logger.setLevel(logging.DEBUG)
        
        # Create and add our handler to root logger
        handler = self._QueueLogHandler(self.log_queue)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        root_logger.addHandler(handler)
        self.active_handlers.append(('root', handler))
        
        # Also specifically capture common library loggers that might be used
        common_loggers = [
            'agent',
            'shipping',
            'browser_use',
            'langchain',
            'openai',
            'google',
            'groq',
            'selenium',
            'requests',
            'urllib3',
            'asyncio',
            'httpx'
        ]
        
        for logger_name in common_loggers:
            try:
                target_logger = logging.getLogger(logger_name)
                if target_logger != root_logger:  # Avoid duplicating root logger
                    self.original_levels[logger_name] = target_logger.level
                    target_logger.setLevel(logging.DEBUG)
                    
                    # Create specific handler for this logger
                    specific_handler = self._QueueLogHandler(self.log_queue)
                    specific_handler.setFormatter(formatter)
                    
                    target_logger.addHandler(specific_handler)
                    self.active_handlers.append((logger_name, specific_handler))
            except Exception as e:
                logger.debug(f"Could not set up logging for {logger_name}: {e}")
        
        # Force enable propagation for all loggers to ensure we catch everything
        for name in logging.Logger.manager.loggerDict:
            try:
                target_logger = logging.getLogger(name)
                target_logger.propagate = True
            except:
                pass
    
    def stop_capture(self):
        """Stop capturing logs and restore original state."""
        if not self.capturing:
            return
            
        # Remove all handlers we added
        for logger_name, handler in self.active_handlers:
            try:
                target_logger = logging.getLogger(logger_name)
                target_logger.removeHandler(handler)
                
                # Restore original level
                if logger_name in self.original_levels:
                    target_logger.setLevel(self.original_levels[logger_name])
            except Exception as e:
                logger.debug(f"Error removing handler from {logger_name}: {e}")
        
        self.active_handlers.clear()
        self.original_levels.clear()
        self.capturing = False
    
    def get_logs(self) -> List[Dict]:
        """Get all logs from the queue."""
        logs = []
        while True:
            try:
                log_entry = self.log_queue.get_nowait()
                logs.append(log_entry)
            except Empty:
                break
        return logs
    
    def has_logs(self) -> bool:
        """Check if there are logs in the queue."""
        return not self.log_queue.empty()


class SimpleShippingChatBot:
    """
    Shipping chat bot with complete terminal log streaming from agent.
    """
    
    def __init__(self, llm_provider: str = None, api_key: str = None):
        """Initialize the chat bot with the shipping tracking agent."""
        # Set API key as environment variable if provided
        if api_key and llm_provider:
            import os
            # Map provider to environment variable name
            env_key_map = {
                'openai': 'OPENAI_API_KEY',
                'groq': 'GROQ_API_KEY', 
                'gemini': 'GOOGLE_API_KEY'
            }
            env_key = env_key_map.get(llm_provider.lower())
            if env_key:
                os.environ[env_key] = api_key
        
        # Initialize agent with better error handling
        self.agent = ShippingTrackingAgent(llm_provider, api_key)
                    
        
        self.booking_patterns = [
            r'\b[A-Z]{4}\d{8,12}\b',  # SINI25432400 pattern
            r'\b[A-Z]{3,5}\d{6,10}\b',  # General booking pattern
            r'\b\d{10,15}\b',  # Pure numeric booking IDs
            r'\b[A-Z0-9]{8,15}\b'  # Mixed alphanumeric
        ]
        
        # Initialize terminal log capture
        self.log_capture = TerminalLogCapture()
    
    def extract_booking_ids(self, text: str) -> List[str]:
        """Extract potential booking IDs from user text."""
        booking_ids = []
        text_upper = text.upper()
        
        for pattern in self.booking_patterns:
            matches = re.findall(pattern, text_upper)
            booking_ids.extend(matches)
        
        return list(dict.fromkeys(booking_ids))

    async def classify_query_with_logs(self, user_input: str, step=None) -> Tuple[str, List[str]]:
        """Use LLM to classify the user query and extract booking IDs with terminal log streaming."""
        if step:
            step.output = "🤖 Analyzing your message with AI..."
            await step.update()
        
        # Start capturing ALL terminal logs
        self.log_capture.start_capture()
        
        classification_prompt = f"""
        Analyze the following user message and classify it. Extract any shipping booking IDs and determine the query type.

        User message: "{user_input}"

        Please respond with a JSON object containing:
        1. "query_type": One of ["tracking_request", "general_question", "greeting", "help_request", "complaint"]
        2. "booking_ids": Array of any booking/container IDs found
        3. "intent": Brief description of what the user wants

        Examples of booking IDs: SINI25432400, ABCD12345678, 1234567890

        Return only valid JSON, no additional text.
        """

        try:
            # Create a task to update logs while processing
            async def update_logs():
                logs_shown = set()
                while True:
                    if self.log_capture.has_logs():
                        new_logs = self.log_capture.get_logs()
                        for log in new_logs:
                            log_id = f"{log['timestamp']}-{log['message']}"
                            if log_id not in logs_shown and step:
                                current_output = step.output or ""
                                emoji = self._get_log_emoji(log['level'])
                                timestamp = log['timestamp'].strftime('%H:%M:%S')
                                logger_name = log.get('logger_name', 'unknown')
                                new_line = f"\n[{timestamp}] {emoji} [{logger_name}] {log['message']}"
                                step.output = current_output + new_line
                                await step.update()
                                logs_shown.add(log_id)
                    await asyncio.sleep(0.1)
            
            # Start log monitoring
            log_task = None
            if step:
                log_task = asyncio.create_task(update_logs())
            
            # Make the actual LLM call
            response = self.agent.llm.invoke(classification_prompt)
            result = response.content.strip()
            
            # Stop log monitoring
            if log_task:
                log_task.cancel()
                try:
                    await log_task
                except asyncio.CancelledError:
                    pass
            
            # Stop capturing logs
            self.log_capture.stop_capture()
            
            # Get any remaining logs
            if step:
                remaining_logs = self.log_capture.get_logs()
                for log in remaining_logs:
                    emoji = self._get_log_emoji(log['level'])
                    timestamp = log['timestamp'].strftime('%H:%M:%S')
                    logger_name = log.get('logger_name', 'unknown')
                    step.output += f"\n[{timestamp}] {emoji} [{logger_name}] {log['message']}"
                await step.update()
            
            # Clean the response
            if result.startswith('```json'):
                result = result.replace('```json', '').replace('```', '').strip()
            
            classification = json.loads(result)
            
            query_type = classification.get('query_type', 'general_question')
            booking_ids = classification.get('booking_ids', [])
            
            if step:
                step.output += f"\n✅ Classification complete: {query_type}"
                if booking_ids:
                    step.output += f"\n🔍 Found booking IDs: {booking_ids}"
                await step.update()
            
            logger.info(f"Query classified as: {query_type}, Booking IDs found: {booking_ids}")
            return query_type, booking_ids
            
        except Exception as e:
            self.log_capture.stop_capture()
            logger.error(f"Error classifying query: {str(e)}")
            # Fallback to simple pattern matching
            booking_ids = self.extract_booking_ids(user_input)
            if booking_ids:
                return "tracking_request", booking_ids
            else:
                return "general_question", []

    async def track_with_terminal_logs(self, booking_ids: List[str], step=None) -> Dict:
        """Track shipments with complete terminal log streaming from agent."""
        if not step:
            if len(booking_ids) == 1:
                return await self.agent.track_shipment(booking_ids[0])
            else:
                return await self.agent.track_multiple_shipments(booking_ids)
        
        # Initialize step display
        step.output = f"🚢 Starting shipment tracking...\n📋 Booking IDs: {booking_ids}\n\n--- Complete Terminal Logs ---"
        await step.update()
        
        # Start capturing ALL terminal logs
        self.log_capture.start_capture()
        
        async def stream_terminal_logs():
            """Stream all terminal logs as they happen."""
            logs_shown = set()
            base_output = f"🚢 Starting shipment tracking...\n📋 Booking IDs: {booking_ids}\n\n--- Complete Terminal Logs ---"
            
            while True:
                if self.log_capture.has_logs():
                    new_logs = self.log_capture.get_logs()
                    for log in new_logs:
                        log_id = f"{log['timestamp']}-{log['logger_name']}-{log['message']}"
                        if log_id not in logs_shown:
                            emoji = self._get_log_emoji(log['level'])
                            timestamp = log['timestamp'].strftime('%H:%M:%S.%f')[:-3]  # Include milliseconds
                            logger_name = log.get('logger_name', 'unknown')
                            
                            # Format the log line similar to terminal output
                            log_line = f"\n[{timestamp}] {emoji} [{logger_name}] {log['level']}: {log['message']}"
                            base_output += log_line
                            logs_shown.add(log_id)
                            
                            step.output = base_output
                            await step.update()
                
                await asyncio.sleep(0.05)  # Check more frequently for better real-time feel
        
        # Start log streaming
        log_stream_task = asyncio.create_task(stream_terminal_logs())
        
        try:
            # Perform actual tracking - this will generate all the terminal logs
            if len(booking_ids) == 1:
                results = await self.agent.track_shipment(booking_ids[0])
            else:
                results = await self.agent.track_multiple_shipments(booking_ids)
        finally:
            # Stop log streaming
            log_stream_task.cancel()
            try:
                await log_stream_task
            except asyncio.CancelledError:
                pass
            
            # Stop log capture
            self.log_capture.stop_capture()
        
        # Get any final logs
        final_logs = self.log_capture.get_logs()
        if final_logs and step:
            for log in final_logs:
                emoji = self._get_log_emoji(log['level'])
                timestamp = log['timestamp'].strftime('%H:%M:%S.%f')[:-3]
                logger_name = log.get('logger_name', 'unknown')
                step.output += f"\n[{timestamp}] {emoji} [{logger_name}] {log['level']}: {log['message']}"
            await step.update()
        
        # Show completion
        if step:
            step.output += f"\n\n✅ Tracking completed successfully!"
            await step.update()
        
        return results

    def _get_log_emoji(self, level: str) -> str:
        """Get emoji for log level."""
        emoji_map = {
            'DEBUG': '🔍',
            'INFO': 'ℹ️',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'CRITICAL': '🚨'
        }
        return emoji_map.get(level, 'ℹ️')

    async def generate_response(self, user_input: str, query_type: str, booking_ids: List[str], tracking_results: Dict = None, step=None) -> str:
        """Generate a natural language response using LLM."""
        if step:
            step.output = "🤖 Generating personalized response..."
            await step.update()
        
        if query_type == "tracking_request" and tracking_results:
            response_prompt = f"""
            Generate a helpful, conversational response for a shipping tracking query.
            
            User asked: "{user_input}"
            Booking IDs tracked: {booking_ids}
            
            Tracking results: {json.dumps(tracking_results, indent=2)}
            
            Create a natural, friendly response that:
            1. Acknowledges their query
            2. Provides the key tracking information clearly
            3. Highlights important dates, status, and vessel info
            4. Uses emojis appropriately for better readability
            5. Offers to help with additional questions
            
            Keep the response conversational and helpful, not robotic.
            """
        else:
            response_prompt = f"""
            Generate a helpful response for this shipping-related query.
            
            User message: "{user_input}"
            Query type: {query_type}
            
            Provide a helpful, conversational response that:
            1. Addresses their question or concern
            2. If they're asking about tracking, explain what booking ID format they should provide
            3. Offer assistance and next steps
            4. Keep it friendly and professional
            5. Use appropriate emojis for better engagement
            
            If they haven't provided a booking ID but want tracking, ask them to provide it.
            """
        
        try:
            # Start capturing logs for response generation too
            if step:
                self.log_capture.start_capture()
                
                async def update_response_logs():
                    while True:
                        if self.log_capture.has_logs():
                            logs = self.log_capture.get_logs()
                            for log in logs:
                                emoji = self._get_log_emoji(log['level'])
                                timestamp = log['timestamp'].strftime('%H:%M:%S')
                                logger_name = log.get('logger_name', 'unknown')
                                step.output += f"\n[{timestamp}] {emoji} [{logger_name}] {log['message']}"
                            await step.update()
                        await asyncio.sleep(0.1)
                
                log_task = asyncio.create_task(update_response_logs())
            
            response = self.agent.llm.invoke(response_prompt)
            
            if step:
                log_task.cancel()
                try:
                    await log_task
                except asyncio.CancelledError:
                    pass
                self.log_capture.stop_capture()
                step.output += "\n✅ Response generated successfully!"
                await step.update()
            
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again or contact support."

    async def process_message(self, user_input: str, step=None) -> Tuple[str, Dict]:
        """Process a user message and return appropriate response with metadata."""
        try:
            if step:
                step.output = "🚀 Starting to process your request..."
                await step.update()
            
            # Classify the query with terminal log streaming
            query_type, booking_ids = await self.classify_query_with_logs(user_input, step)
            
            tracking_results = None
            
            # If it's a tracking request with booking IDs, perform tracking with terminal logs
            if query_type == "tracking_request" and booking_ids:
                tracking_results = await self.track_with_terminal_logs(booking_ids, step)
            
            # Generate natural language response
            response = await self.generate_response(user_input, query_type, booking_ids, tracking_results, step)
            
            # Prepare metadata
            metadata = {
                "query_type": query_type,
                "booking_ids": booking_ids,
                "tracking_results": tracking_results,
                "timestamp": datetime.now().isoformat()
            }
            
            if step:
                step.output += "\n\n🎉 All processing completed successfully!"
                await step.update()
            
            return response, metadata
            
        except Exception as e:
            error_msg = f"I apologize, but I encountered an error processing your request: {str(e)}"
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            
            if step:
                step.output += f"\n❌ Error occurred: {str(e)}"
                await step.update()
            
            return error_msg, {"error": str(e)}


# LLM Configuration Settings
LLM_PROVIDERS = {
    "openai": {
        "name": "OpenAI GPT",
        "models": ["gpt-4", "gpt-3.5-turbo"],
        "key_placeholder": "sk-..."
    },
    "groq": {
        "name": "Groq",
        "models": ["mixtral-8x7b-32768", "llama2-70b-4096"],
        "key_placeholder": "gsk_..."
    },
    "gemini": {
        "name": "Google Gemini",
        "models": ["gemini-pro", "gemini-pro-vision"],
        "key_placeholder": "AI..."
    }
}


async def setup_llm_configuration():
    """Setup LLM configuration through UI."""
    # Set up the configuration form
    settings = await cl.ChatSettings([
        cl.input_widget.Select(
            id="llm_provider",
            label="🤖 Select AI Provider",
            values=list(LLM_PROVIDERS.keys()),
            initial_index=0,
        ),
        cl.input_widget.TextInput(
            id="api_key",
            label="🔑 API Key",
            placeholder="Enter your API key here...",
        ),
        cl.input_widget.Switch(
            id="show_detailed_logs",
            label="📊 Show Complete Terminal Logs",
            initial=True,
        ),
    ]).send()
    
    return settings


def validate_api_key(provider: str, api_key: str) -> bool:
    """Validate API key format for different providers."""
    if not api_key:
        return False
    
    # Basic validation - just check if key is non-empty and reasonable length
    if len(api_key) < 10:
        return False
    
    validation_patterns = {
        "openai": r'^sk-[a-zA-Z0-9]{48,}$',
        "groq": r'^gsk_[a-zA-Z0-9_]{52,}$',
        "gemini": r'^AI[a-zA-Z0-9]{35,}$'
    }

    pattern = validation_patterns.get(provider)
    if pattern:
        import re
        return bool(re.match(pattern, api_key))
    
    return True  # Basic length check for unknown providers


# Chainlit event handlers
@cl.on_chat_start
async def start():
    """Initialize the chat session with LLM configuration."""
    try:
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        
        # Set up logging to capture more details
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Show configuration prompt
        await cl.Message(content="""# 🚢 Shipping Tracking Assistant Setup

Welcome! This version features **complete terminal log streaming** from your ShippingTrackingAgent.

**Enhanced Features:**
- 📡 **Complete Terminal Log Streaming** - See ALL logs from your agent and dependencies as they appear in terminal
- 🔍 **Live Progress Tracking** - Watch every step of your agent's execution in real-time
- 📊 **Full Logging Context** - Capture logs from all libraries (browser_use, langchain, requests, etc.)
- ⚡ **Real-time Updates** - Terminal-style log streaming with timestamps and logger names

**What You'll See:**
- All `logging` statements from your agent
- Dependency logs (browser automation, HTTP requests, etc.)
- Error traces and debug information
- Exactly what appears in your terminal when running the agent

**Supported Providers:**
- 🤖 **OpenAI GPT** - High-quality responses with GPT-4 or GPT-3.5
- ⚡ **Groq** - Fast inference with Mixtral and Llama models  
- 🧠 **Google Gemini** - Google's latest AI models

Please use the settings panel to configure your preferred AI provider and API key.
""").send()
        
        # Setup configuration
        settings = await setup_llm_configuration()
        
        # Store initial settings
        cl.user_session.set("settings", settings)
        
    except Exception as e:
        error_msg = f"❌ Sorry, there was an error during setup: {str(e)}"
        await cl.Message(content=error_msg).send()
        logger.error(f"Chat setup error: {e}", exc_info=True)


@cl.on_settings_update
async def setup_agent(settings):
    """Handle settings updates and initialize the agent."""
    try:
        print(f"SETTINGS: {settings}")
        llm_provider = settings.get("llm_provider", "gemini")
        api_key = settings.get("api_key", "")
        show_logs = settings.get("show_detailed_logs", True)
        
        print(f"API KEY: {api_key}")
        
        # Show initialization progress
        init_msg = await cl.Message(content=f"🚀 Initializing with {LLM_PROVIDERS[llm_provider]['name']} and setting up complete terminal log streaming...").send()
        
        try:
            # Initialize the chatbot with selected provider and API key
            chatbot = SimpleShippingChatBot(llm_provider, api_key)
            
            # Test the connection
            try:
                test_response = chatbot.agent.llm.invoke("Hello, please respond with 'Connection successful!'")
                logger.info(f"Connection test successful: {test_response.content[:50]}...")
            except Exception as test_error:
                logger.warning(f"Connection test failed, but continuing: {test_error}")
            
            # Store in user session
            cl.user_session.set("chatbot", chatbot)
            cl.user_session.set("settings", settings)
            
            # Get LLM info safely
            try:
                if hasattr(chatbot.agent, 'get_llm_info'):
                    llm_info = chatbot.agent.get_llm_info()
                    provider_display = f"{llm_info['provider']} ({llm_info['model']})"
                else:
                    provider_display = f"{LLM_PROVIDERS[llm_provider]['name']}"
            except:
                provider_display = f"{LLM_PROVIDERS[llm_provider]['name']}"
            
            # Send success message
            success_msg = f"""✅ **Successfully Connected with Complete Terminal Log Streaming!**

🤖 **AI Provider**: {provider_display}
📊 **Terminal Logs**: {'Enabled - All logs captured' if show_logs else 'Disabled'}
🔄 **Log Sources**: Root logger + all dependencies (browser_use, langchain, requests, etc.)

---

# 🚢 Shipping Tracking Assistant

I'm ready to help you track shipments with **complete terminal log streaming**!

**What you'll see in real-time:**
- 📡 All logging output that appears in your terminal
- 🔍 Logs from browser automation (browser_use)
- 📊 HTTP requests and API calls
- ⚡ Debug information and error traces
- 🕐 Timestamps and logger names for each log entry

**Log Sources Captured:**
- Your agent's logging statements
- browser_use library logs
- langchain/LLM provider logs
- HTTP request logs (requests, urllib3)
- Any other dependency logs

*Examples:*
- "Track SINI25432400"
- "Where is my shipment ABCD12345678?"
- "What's the status of booking 1234567890?"

Try a tracking request to see the complete terminal logs in action!
"""
            
            await init_msg.remove()
            await cl.Message(content=success_msg).send()
            
        except Exception as e:
            error_details = str(e)
            logger.error(f"Agent initialization error: {error_details}", exc_info=True)
            
            error_msg = f"""❌ **Initialization Failed**

Error: `{error_details}`

**Troubleshooting:**
1. **Agent Setup**: Ensure your `ShippingTrackingAgent` class can accept `llm_provider` and `api_key` parameters
2. **Logging Setup**: Make sure your agent uses Python's `logging` module
3. **Dependencies**: Check that all required packages are installed

**For Complete Terminal Logging:** Your agent should include logging statements like:
```python
import logging
logger = logging.getLogger(__name__)

class ShippingTrackingAgent:
    def track_shipment(self, booking_id):
        logger.info(f"Starting tracking for")
        logger.debug("Connecting to shipping API...")
        # Your tracking logic here
        logger.info("Tracking completed successfully")
```

**Captured Dependencies:**
- browser_use (browser automation logs)
- langchain (LLM interaction logs)  
- requests/urllib3 (HTTP request logs)
- All other standard library logging
"""
            
            await init_msg.remove()
            await cl.Message(content=error_msg).send()
            
    except Exception as e:
        error_msg = f"❌ **Setup Error**: {str(e)}"
        await cl.Message(content=error_msg).send()
        logger.error(f"Settings update error: {e}", exc_info=True)


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages with complete terminal log streaming."""
    try:
        # Get the chatbot from user session
        chatbot = cl.user_session.get("chatbot")
        settings = cl.user_session.get("settings", {})
        
        if not chatbot:
            await cl.Message(content="""❌ **Not Configured**

Please configure your AI provider settings using the settings panel before sending messages.

Click the ⚙️ settings icon to get started!""").send()
            return
        
        show_logs = settings.get("show_detailed_logs", True)
        
        # Process with complete terminal log streaming if enabled
        if show_logs:
            async with cl.Step(name="🔄 Processing with Complete Terminal Log Stream") as step:
                response, metadata = await chatbot.process_message(message.content, step)
        else:
            # Process without detailed logs
            response, metadata = await chatbot.process_message(message.content)
        
        # Create response message
        msg = cl.Message(content=response)
        
        
        # Add tracking results as element if available
        if metadata.get("tracking_results") and metadata.get("booking_ids"):
            tracking_data = metadata["tracking_results"]
            
            if isinstance(tracking_data, dict) and not tracking_data.get('error'):
                tracking_info = json.dumps(tracking_data, indent=2)
                element = cl.Text(
                    name="📋 Detailed Tracking Results",
                    content=tracking_info,
                    display="side"
                )
                msg.elements = [element]
        
        await msg.send()
        
    except Exception as e:
        logger.error(f"Error in main message handler: {e}", exc_info=True)
        error_response = f"❌ An unexpected error occurred: {str(e)}"
        await cl.Message(content=error_response).send()


@cl.on_stop
async def stop():
    """Handle session end."""
    logger.info("Chat session ended")


if __name__ == "__main__":
    pass