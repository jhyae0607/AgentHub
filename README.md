# AgentHub

## Overview

A comprehensive application for building and running AI agent services. It extends Agent Service Toolkit (https://github.com/JoshuaC215/agent-service-toolkit?tab=readme-ov-file).

The application offers a template to easily run LangGraph agents.

### Quickstart

Run directly in python

```sh
# At least one LLM API key is required or Ollama is available as well
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env

# uv is recommended but "pip install ." also works
pip install uv
uv sync --frozen
# "uv sync" creates .venv automatically
source .venv/bin/activate
python src/run_service.py

# In another shell
source .venv/bin/activate
streamlit run src/streamlit_app.py
```

### TODO List:

- Reframe outputs of the agents to be more user friendly.
- Agents to create: Education Assistant, MCP Server Coder
- Save/export chat history
- Implement LlamaGuard to limit harmful responses
- More sophisticated parsing technique (Unsturctured/Docling vs VLMs)
- Text-to-Audio capability
- UI Improvements
- Writing Impact Paper

### Key Files

The repository is structured as follows:

- `src/agents/`: Defines several agents with different capabilities
- `src/schema/`: Defines the protocol schema
- `src/core/`: Core modules including LLM definition and settings
- `src/service/service.py`: FastAPI service to serve the agents
- `src/client/client.py`: Client to interact with the agent service
- `src/streamlit_app.py`: Streamlit app providing a chat interface
- `tests/`: Unit and integration tests

### Local development without Docker

You can also run the agent service and the Streamlit app locally without Docker, just using a Python virtual environment.

1. Create a virtual environment and install dependencies:

   ```sh
   pip install uv
   uv sync --frozen
   source .venv/bin/activate
   ```

2. Run the FastAPI server:

   ```sh
   python src/run_service.py
   ```

3. In a separate terminal, run the Streamlit app:

   ```sh
   streamlit run src/streamlit_app.py
   ```

4. Open your browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).
