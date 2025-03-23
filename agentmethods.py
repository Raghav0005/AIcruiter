import asyncio
import json
import os
import sys
import time
import ssl
import certifi

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from rag import get_rag_content
import google.generativeai as genai

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.google import GoogleLLMContext, GoogleLLMService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.tavus import TavusVideoService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from google.ai.generativelanguage_v1beta.types.content import Content
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

load_dotenv(override=True)
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

ssl_context = ssl.create_default_context(cafile=certifi.where())

# Import the email sender
from email_sender import send_email

### Step 2: Load candidate details (from JSON) ###
def load_candidate_details():
    with open('details.json', 'r') as f:
        return json.load(f)

### Step 4: Update prompts using candidate details ###
def update_prompts(candidate_details):
    # Update RAG content and prompt using candidate resume details
    resume_filename = candidate_details.get('resume', '')
    rag_content = get_rag_content(os.path.join('uploads', resume_filename))
    
    RAG_PROMPT = f"""
    You are an interviewer designed to interview a candidate based solely on the provided knowledge base, which is a resume.

    **Instructions:**

    1.  **Knowledge Base Only:** Answer questions *exclusively* using the information in the "Knowledge Base" section below. Do not use any outside information.
    2.  **Conversation History:** Use the "Conversation History" (ordered oldest to newest) to understand the context of the current question.
    3.  **Concise Response:**  Respond in 50 words or fewer.  The response will be spoken, so avoid symbols, abbreviations, or complex formatting. Use plain, natural language.
    4. Just provide the follow-up response or question to ask the interviewee.
    5. You must follow all instructions.

    **Input Format:**

    Each request will include:

    *   **Conversation History:**  (A list of previous user and assistant messages, if any)

    **Knowledge Base:**
    Here is the knowledge base you have access to:
    {rag_content}
    """
    
    candidate_name = candidate_details.get('name', 'Candidate')
    candidate_links = candidate_details.get('links', [])
    system_prompt = f"""You are conducting a behavioural interview for {candidate_name} on behalf of Google.
    Initially start the interview with introductions, and small questions for a little bit.
    You have access to the candidate's resume with the function query_knowledge_database. You can use the function
    to ask any question about his/her resume, and you should a couple times in the interview, up to your discretion.
    You can also query it if a follow-up question regarding the resume might be helpful at the point in the interview.
    Your output will be converted to audio so don't include special characters in your answers.
    Additional candidate links: {', '.join(candidate_links)}
    """
    
    return RAG_PROMPT, system_prompt

### Step 3: Initialize the room and send the URL via email ###
async def initialize_room(session):
    tavus = TavusVideoService(
        api_key="3c18d259a55f4786a94b948f97004afd",
        replica_id="raff7843cc3d",
        session=session,
        sample_rate=16000,
    )
    persona_name = await tavus.get_persona_name()
    room_url = await tavus.initialize()
    # Here you would call your email function to send the room_url with candidate details.
    # e.g., send_email(candidate_details, room_url)
    logger.info(f"Room URL is {room_url} and will be emailed to the candidate.")
    return tavus, persona_name, room_url

def send_interview_email(candidate_details, room_url):
    """Send the interview URL to the candidate via email"""
    recipient_email = candidate_details.get('email', '')
    name = candidate_details.get('name', 'Candidate')
    subject = f"Your AIcruiter Interview Link - {name}"
    body = f"""Hello {name},

Thank you for your interest in our position. We've prepared an AI-powered interview for you.

Please click the link below to start your interview:
{room_url}

The interview will assess your qualifications and experience. You can take it anytime within the next 48 hours.

Best regards,
The Hiring Team
"""
    
    # Send the email
    smtp_server = 'smtp.gmail.com'
    port = 465
    sender_email = 'vasuraghav04@gmail.com'
    sender_password = 'utgn zjdp tyhs udnc'
    
    send_email(smtp_server, port, sender_email, sender_password, recipient_email, subject, body)
    logger.info(f"Interview email sent to {recipient_email}")

### The query_knowledge_base function remains separate ###
async def query_knowledge_base(RAG_PROMPT, function_name, tool_call_id, arguments, llm, context, result_callback):
    logger.info(f"Querying knowledge base for question: {arguments['question']}")
    client = genai.GenerativeModel(
        model_name="gemini-2.0-flash-lite-preview-02-05",
        system_instruction=RAG_PROMPT,
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=64,
        ),
    )
    # Process conversation history
    conversation_turns = context.messages[2:]
    messages = []
    for turn in conversation_turns:
        messages.extend(context.to_standard_messages(turn))

    messages = [turn for turn in messages if turn.get("role") != "tool" and not turn.get("tool_calls")]
    messages = messages[-3:]
    messages_json = json.dumps(messages, ensure_ascii=False, indent=2)

    logger.info(f"Conversation turns: {messages_json}")
    start = time.perf_counter()
    response = client.generate_content(contents=[messages_json])
    end = time.perf_counter()
    logger.info(f"Time taken: {end - start:.2f} seconds")
    logger.info(response.text)
    await result_callback(response.text)

### Step 5: Run the LLM pipeline ###
async def run_pipeline(session, system_prompt, room_url, tavus, persona_name):
    # Set up services: STT, TTS, LLM
    stt = DeepgramSTTService(api_key="d5a636d2ac4bf7071df5e90bd0131213a27eeb81")
    tts = CartesiaTTSService(
        api_key="sk_car_b1CdV89DpHq0njLlsWijO",
        voice_id="58db94c7-8a77-46a7-9107-b8b957f164a0",
    )
    llm = GoogleLLMService(
        api_key='AIzaSyAQRHz9zd9JX0PXx80TxAO8oBmFvarYVyo',
        model="gemini-2.0-flash-001"
    )
    llm.register_function("query_knowledge_base", query_knowledge_base)
    tools = [{
        "function_declarations": [
            {
                "name": "query_knowledge_base",
                "description": "Query the knowledge base for the answer to the question.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to query the knowledge base with.",
                        },
                    },
                },
            },
        ],
    }]

    # Create context with the updated system prompt
    messages = [{"role": "system", "content": system_prompt}]
    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

    # Configure the pipeline with all the services and transport
    transport = DailyTransport(
        room_url=room_url,
        token=None,
        bot_name=persona_name,
        params=DailyParams(
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        ),
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        tts,
        context_aggregator.assistant(),
        tavus,
        transport.output(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            allow_interruptions=True,
            enable_metrics=False,
            enable_usage_metrics=False,
            report_only_initial_ttfb=True,
        ),
    )

    # (Optional) Define event handlers for participants if needed
    @transport.event_handler("on_first_participant_joined")
    async def on_participant_joined(transport: DailyTransport, participant: dict) -> None:
        logger.debug(f"Ignoring {participant['id']}'s microphone")
        await transport.update_subscriptions({
            participant["id"]: {"media": {"microphone": "unsubscribed"}}
        })
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        await task.cancel()

    runner = PipelineRunner()
    await runner.run(task)

### Main orchestration function ###
async def process_interview_request():
    """Main function that orchestrates the entire interview process"""
    # 1. Load candidate details from JSON
    candidate_details = load_candidate_details()
    
    # 2. Initialize an aiohttp session for HTTP requests
    async with aiohttp.ClientSession() as session:
        try:
            # 3. Initialize the room and get the URL
            logger.info("Initializing interview room...")
            tavus, persona_name, room_url = await initialize_room(session)
            
            # 4. Send the URL to the candidate via email
            logger.info(f"Sending interview link to candidate: {candidate_details.get('name', 'Candidate')}")
            send_interview_email(candidate_details, room_url)
            
            # 5. Update the prompts with candidate details
            logger.info("Updating prompts with candidate details...")
            RAG_PROMPT, system_prompt = update_prompts(candidate_details)
            
            # 6. Run the LLM pipeline
            logger.info("Starting interview pipeline...")
            await run_pipeline(session, system_prompt, room_url, tavus, persona_name)
            
            return True, "Interview process completed successfully"
        except Exception as e:
            logger.error(f"Error in interview process: {str(e)}")
            return False, str(e)

def start_interview_process():
    """Function to be called from Flask to start the interview process"""
    # Run the async process in a new event loop
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success, message = loop.run_until_complete(process_interview_request())
        return success, message
    except Exception as e:
        logger.error(f"Error in interview process: {e}")
        return False, str(e)

# Allow direct execution for testing
if __name__ == "__main__":
    success, message = start_interview_process()
    print(f"Process {'succeeded' if success else 'failed'}: {message}")