<h1 align="center">
ZoomGPT:🎙️Ask me anything
</h1>

Talk to your Zoom recorded meeting transcripts.

## Features

- Upload transcript (VTT format) and ask questions about the meeting.
- Responses infused with emojis for added flare.

## Install modules

```
pip install -U openai tiktoken streamlit anthropic
```

## Set environment variables in .env

For OpenAI

```
OPENAI_API_KEY=XXX
DEPLOYMENT_NAME="gpt-4-32k"
MAX_TOKEN_VALUE=32768
```

For Azure OpenAI

```
OPENAI_API_KEY=XXX
OPENAI_API_BASE=XXX
OPENAI_API_TYPE=XXX
OPENAI_API_VERSION=XXX
DEPLOYMENT_NAME="gpt-4-32k"
MAX_TOKEN_VALUE=32768
```

For Anthropic Claude

```
ANTHROPIC_API_KEY=XXX
DEPLOYMENT_NAME="claude-2"
MAX_TOKEN_VALUE=102400
```

## Run

```
streamlit run zoomgpt.py
```