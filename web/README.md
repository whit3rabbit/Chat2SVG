# Chat2SVG Web Interface

This is the web interface for Chat2SVG, allowing you to generate vector graphics using large language models and image diffusion models.

## Features

- Generate SVG templates with Large Language Models
- Enhance details with image diffusion models
- Optimize SVG shapes
- Support for multiple LLM providers:
  - OpenAI
  - Anthropic (Claude)
  - OpenRouter (access to many models)
  - Local LLMs with OpenAI API compatibility

## Setup

### Prerequisites

- Node.js (16+)
- Python 3.10+
- Required dependencies installed (see main README)

### Installation

1. Install backend dependencies:

```bash
cd backend
pip install -r requirements.txt
```

2. Install frontend dependencies:

```bash
cd ../
npm install
```

### Configuration

Create a `.env` file in the project root with your API keys:

```
# OpenAI Settings
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o

# Anthropic (Claude) Settings
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL=claude-3-5-sonnet-20240620

# OpenRouter Settings
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=openai/gpt-4o

# Local LLM Settings
LOCAL_LLM_BASE_URL=http://localhost:1234/v1
LOCAL_LLM_MODEL=default
```

You can also configure these settings through the web interface.

### Running the Application

1. Start the backend server:

```bash
cd backend
uvicorn main:app --reload --port 8000
```

2. Start the frontend development server:

```bash
cd ../
npm start
```

3. Open [http://localhost:3000](http://localhost:3000) in your browser

## Using Local LLMs

Chat2SVG supports any local LLM that exposes an OpenAI-compatible API. Some options include:

- [LM Studio](https://lmstudio.ai/)
- [Ollama](https://ollama.ai/)
- [vLLM](https://github.com/vllm-project/vllm)
- [LocalAI](https://github.com/go-skynet/LocalAI)

Configure your local LLM server to run on a specific port and set the `LOCAL_LLM_BASE_URL` to point to your server.

## Using OpenRouter

[OpenRouter](https://openrouter.ai) provides access to a wide range of LLM models through a single API. Some advantages:

- Access to models from OpenAI, Anthropic, Mistral, Google, and more
- Automatic fallbacks for reliability
- Optimized cost routing

Sign up for an API key at [openrouter.ai](https://openrouter.ai) and configure it in your settings.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 