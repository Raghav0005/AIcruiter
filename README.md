# AIcruiter

Created at GenAI Genesis 2025.

AIcruiter is an AI-powered interview agent that automates and enhances the recruitment process through intelligent candidate screening and interviewing.
It aims to encourage human empowerment by eliminating the unconscious human bias in interviews, along with the time consuming HR work.

## Overview

AIcruiter uses artificial intelligence converstational agents to conduct technical interviews, assess candidate skills, and help recruiters make data-driven hiring decisions. The HR can access our webapp in order to customize the AI bot to their company's values and needs. As they send out their candidate interview request, it sends an email with a link to a WebRTC call so that the candidate can join at their convenience to get interviewed by the bot.

## Tech Stack

- **Backend Framework**: Flask (Python)
- **AI/ML**: 
  - Google Generative AI (Gemini 2.0 Flash)
  - RAG (Retrieval Augmented Generation)
- **Speech Processing**:
  - Deepgram (Speech-to-Text)
  - Cartesia (Text-to-Speech)
- **Frontend**: 
  - HTML/CSS/JavaScript
  - Bootstrap 5 & Bootstrap Icons
  - Jinja2 Templates
- **Media Processing**:
  - Pipecat (Audio/Video Pipeline)
  - Tavus (Video Service)
  - Daily (Transport Service)
- **Document Processing**: PyPDF2 (Resume parsing)
- **Additional Libraries**:
  - aiohttp (Async HTTP)
  - Loguru (Logging)

## Features

- HR Control Center to customize AI agent personality, values, and questions
- Conversational AI Agent that conducts live interviews with real-time STT → LLM → TTS pipeline
- Email Automation to send interview invites with secure video links
- WebRTC-based Interviews using Daily + Pipecat for minimal latency
- Scoring Engine Direct Integration (Coming Soon): Semantic model to rate candidates based on responses

## Installation

1. Clone this repository
   ```
   git clone https://github.com/yourusername/AIcruiter.git
   cd AIcruiter
   ```

2. Create and Activate a Python Virtual Environment
   ```
   python3 -m venv AIcruiter
   source AIcruiter/bin/activate
   ```

3. Install the required dependencies
   ```
   pip install -r requirements.txt
   pip install git+https://github.com/pipecat-ai/pipecat.git
   ```

4. Setup the following variables in a `keys.json` file at the root directory.
Note the `email` and `email_password` are for the invidiuals who send the email to the interviewee.
`email_password` is the app password for the desired email account.
```
{
  "google_gemini_api_key": "",
  "tavus_api_key": "",
  "tavus_replica_id": "",
  "deepgram_api_key": "",
  "cartesia_api_key": "",
  "cartesia_voice_id": "",
  "email": "",
  "email_password": ""
}
```

## Usage

Run the app with `python app.py` or `python3 app.py`, whichever one suits your environment the most.

## Dependencies
Please see dependencies specified in `requirements.txt`.
