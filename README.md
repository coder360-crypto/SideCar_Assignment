# AI-Based Shipping Line Tracking System

A robust, repeatable AI-driven solution that automatically retrieves voyage numbers and arrival dates for HMM shipping line booking IDs using natural language processing and browser automation. This system includes both a command-line interface for direct tracking and an interactive chat application.

## 🎯 Assignment Overview

**Assignment ID**: SINI25432400 (example)
**Objective**: Develop an AI-powered system to extract shipping information from seacargotracking.net (and potentially other sources) without hardcoded web interactions, accessible via CLI and a chat interface.

## ✨ Features

- **Natural Language Processing**: Uses AI to interact with websites through natural language commands.
- **Zero Hardcoding**: Aims to minimize hardcoded selectors or interaction patterns.
- **Adaptive Navigation**: Designed to adapt to website changes and different layouts.
- **Data Persistence**: Stores successful interaction patterns for improved performance (for the agent).
- **Cross-Platform**: Run scripts provided for Windows, macOS, and Linux.
- **Interactive Chat Interface**: A user-friendly web UI (via Chainlit) for tracking shipments and configuring LLM providers.
- **Multi-LLM Support**: Supports various LLM providers (e.g., OpenAI, Groq, Google Gemini) configurable via the chat UI or environment variables.
- **Robust Error Handling**: Comprehensive error handling and retry mechanisms.
- **Real-time Log Streaming**: The chat interface can display real-time logs from the agent.

## 🏗️ System Architecture

**CLI Execution:**
```
User (CLI) --> run.sh / run.bat --> agent.py --> AI Agent Logic --> Browser Automation --> Target Website
                                       ^ |
                                       | v
                                  Data Storage (Cache)
```

**Chat Application Execution:**
```
User (Web Browser) --> Chainlit UI (chat.py) --> SimpleShippingChatBot --+
                                                                         |
                                                                         v
                                                            ShippingTrackingAgent (agent.py)
                                                                         |
                                                                         v
                                                                AI Agent Logic
                                                                         |
                                                                         v
                                                               Browser Automation
                                                                         |
                                                                         v
                                                                  Target Website
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ (Python 3.11 recommended)
- `uv` (recommended for environment and package management) or `pip` and `venv`
- Internet connection
- API key for your chosen LLM provider (e.g., OpenAI, Groq, Google Gemini)

### Installation

1.  **Clone or download the project files.**
    If you have git:
    ```bash
    git clone <repository-url>
    cd <project-directory>
    ```

2.  **Set up Python environment.** (Using `uv` is recommended)

    **Using `uv` (Recommended):**
    ```bash
    # Install uv if you haven't already: pip install uv
    uv venv --python 3.11  # Creates a .venv directory
    # Activate environment:
    # On Windows: .venv\Scripts\activate
    # On Mac/Linux: source .venv/bin/activate
    ```

    **Using standard `venv`:**
    ```bash
    python3 -m venv venv
    # On Windows: venv\Scripts\activate
    # On Mac/Linux: source venv/bin/activate
    ```

3.  **Install dependencies.**
    ```bash
    # Using uv (fastest)
    uv pip install -r requirements.txt

    # Or using pip
    pip install -r requirements.txt

    # Install Playwright browsers (required for browser automation)
    playwright install
    ```

4.  **Configure environment variables.**
    Copy the example environment file:
    ```bash
    cp .env.example .env
    ```
    Edit the `.env` file and add your API key(s) for the desired LLM provider(s). For example:
    ```env
    # For OpenAI
    OPENAI_API_KEY=your_openai_sk_key_here

    # For Groq
    GROQ_API_KEY=your_groq_gsk_key_here

    # For Google Gemini
    GOOGLE_API_KEY=your_google_ai_key_here

    # You can also set a default LLM_PROVIDER (openai, groq, gemini)
    # LLM_PROVIDER=groq
    ```
    The applications (`agent.py` and `chat.py`) will use these keys. The chat interface also allows setting these at runtime.

## 💻 Usage

There are two main ways to use this system:

### 1. Command-Line Tracking Agent (`agent.py`)

This is for direct, scripted tracking of booking IDs.

**Using the run scripts (Recommended for CLI):**
   ```bash
   # On Windows:
   run.bat SINI25432400

   # On Mac/Linux (make sure it's executable: chmod +x run.sh):
   ./run.sh SINI25432400
   ```
   The script will guide you if the `.env` file is not configured.

**Direct Python execution (CLI):**
   Make sure your virtual environment is activated and `.env` is configured.
   ```bash
   python agent.py SINI25432400
   # For multiple booking IDs:
   python agent.py SINI25432400 ABCD12345678
   ```

**Expected CLI Output (Example):**
   ```
   🚢 Starting shipment tracking for booking ID: SINI25432400
   🤖 Using Groq (llama-4-scout-17b-16e-instruct) # Example LLM
   ✅ Successfully retrieved shipping information!

   ==================================================
   TRACKING RESULTS
   ==================================================
   📦 Booking ID: SINI25432400
   🚢 Voyage Number: HMM001E
   📅 Arrival Date: 2025-06-15
   🏭 Vessel Name: HMM COPENHAGEN
   🔄 Status: In Transit
   🤖 LLM Provider: groq

   ✅ Tracking completed!
   ```

### 2. Interactive Chat Application (`chat.py`)

This provides a web-based UI for tracking shipments, configuring LLM providers, and viewing real-time agent logs.

**Running the chat application:**
   Make sure your virtual environment is activated.
   ```bash
   chainlit run chat.py -w
   ```
   The `-w` flag enables auto-reloading when you make changes to `chat.py`.
   Open your web browser and navigate to the URL provided by Chainlit (usually `http://localhost:8000`).

**Using the Chat Application:**
1.  On the first run, or by clicking the settings (⚙️) icon, configure your LLM Provider and API Key.
2.  Type your tracking request into the chat input (e.g., "Track SINI25432400").
3.  Observe the real-time logs and the agent's progress.
4.  Receive a formatted response with the tracking details.

## ⚙️ Configuration Options

Beyond API keys, other settings can be configured in the `.env` file (primarily for `agent.py`'s direct use or as defaults):

```env
# Browser settings for agent.py when run directly
BROWSER_HEADLESS=true          # Set to false to see browser window
BROWSER_TIMEOUT=30000          # Timeout in milliseconds
BROWSER_SLOW_MO=100            # Slow down automation for debugging

# Caching for agent.py
CACHE_EXPIRY_HOURS=24          # How long to cache results

# Retry settings for agent.py
MAX_RETRY_ATTEMPTS=3           # Number of retry attempts
RETRY_DELAY_SECONDS=5          # Delay between retries

# Debugging for agent.py
ENABLE_SCREENSHOTS=false       # Save screenshots for debugging agent.py
LOG_LEVEL=INFO                 # Logging level (DEBUG, INFO, WARNING, ERROR) for agent.py
LOG_FILE=shipping_tracking.log # Log file for agent.py direct runs
```
The `chat.py` application has its own logging and settings UI for some of these.

## 🔧 Advanced Usage & Customization

### Batch Processing with `agent.py`

You can script `agent.py` for batch processing multiple booking IDs. Refer to the `agent.py`'s `main()` function or create a custom script that imports `ShippingTrackingAgent`.

### Supporting Different Booking IDs / Shipping Lines

- **Booking IDs**: The AI agent is designed to be flexible. The prompts in `agent.py` can be adjusted to better guide the AI for specific ID formats or websites.
- **Shipping Lines**: To support new shipping lines:
    1.  Update the AI prompts in `agent.py` (e.g., `tracking_prompt`) to include instructions for the new line/website.
    2.  The `Config` class (`config.py`) could be extended if specific URLs or patterns for new lines need to be managed systemically, though the current approach relies more on AI flexibility.

## 🧪 Output Verification

1.  **Console Output (CLI)**: Real-time status updates and results from `agent.py`.
2.  **Chat Interface (UI)**: Formatted responses and detailed tracking results viewable in the UI.
3.  **Real-time Logs (UI)**: The chat interface streams logs from the agent.
4.  **Database Storage (`shipping_tracking.db`)**: `agent.py` can cache results.
5.  **Screenshots**: If `ENABLE_SCREENSHOTS=true` in `.env`, `agent.py` may save screenshots in a `screenshots/` directory during debugging.
6.  **Log File**: For CLI runs of `agent.py`, logs are stored in `shipping_tracking.log` (or as configured).

## 🔄 Adaptability and Generalization

The system's adaptability relies on:
- **LLM Capabilities**: The core of understanding and navigating websites is handled by the LLM.
- **Flexible Prompts**: The prompts given to the AI in `agent.py` are crucial. They guide the AI without hardcoding specific steps.
- **Browser Automation Tools**: `browser_use` library abstracts many browser interaction complexities.
- **Error Handling and Fallbacks**: `agent.py` includes logic for retries and potential fallback strategies.

## 🛠️ Troubleshooting

### Common Issues

1.  **API Key Not Found/Invalid**:
    *   Ensure your `.env` file is correctly formatted and contains the valid API key for the selected LLM provider.
    *   In the chat app, double-check the API key entered in the settings panel.
    *   Verify the environment variable name matches what the application expects (e.g., `OPENAI_API_KEY`, `GROQ_API_KEY`).

2.  **"Browser launch failed" / Playwright Issues**:
    *   Ensure Playwright browsers are installed: `playwright install`.
    *   Try running with `BROWSER_HEADLESS=false` (in `.env` for `agent.py`) to see if the browser window provides more clues.

3.  **"No tracking results found" / AI Agent Issues**:
    *   The target website might have changed. Try adjusting the prompt in `agent.py` (`tracking_prompt`).
    *   The booking ID might be invalid or not in the system.
    *   Check `LOG_LEVEL=DEBUG` in `.env` for more detailed logs from `agent.py`.
    *   In the chat app, enable "Show Real-time Agent Logs" for insights.

4.  **Module Not Found Errors**:
    *   Ensure your virtual environment is activated.
    *   Reinstall dependencies: `uv pip install -r requirements.txt` or `pip install -r requirements.txt`.

5.  **Chainlit App Not Starting / UI Issues**:
    *   Check the terminal output when you run `chainlit run chat.py -w` for any error messages.
    *   Ensure Chainlit is installed correctly in your virtual environment.
    *   Try accessing in a different browser or incognito mode.

### Debug Mode

- **For `agent.py` (CLI)**:
  Set in `.env`: `BROWSER_HEADLESS=false`, `ENABLE_SCREENSHOTS=true`, `LOG_LEVEL=DEBUG`.
- **For `chat.py` (UI)**:
  Use the "Show Real-time Agent Logs" switch in the chat settings. Examine terminal output from `chainlit run ...`.

## 🎓 Educational Notes

### Key Technologies Used

- **`agent.py`**:
    - `browser_use`: AI-powered browser automation.
    - LangChain (`langchain-openai`, `langchain-google-genai`, `langchain-groq`): LLM integration and orchestration.
    - Playwright: Underlying browser automation engine.
    - Dotenv: Environment variable management.
- **`chat.py`**:
    - Chainlit: Python framework for creating chat UIs.
    - Standard Python libraries (asyncio, logging, json, re).
- **LLMs**: OpenAI GPT models, Groq's Llama/Mixtral models, Google Gemini models.
- **SQLite**: Local data persistence for caching by `agent.py`.

---

This README provides a comprehensive guide to setting up, running, and troubleshooting the AI-Based Shipping Line Tracking System.