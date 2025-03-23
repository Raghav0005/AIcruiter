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
from config import load_api_keys

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

from email_sender import send_email

def load_candidate_details():
    with open('data/details.json', 'r') as f:
        return json.load(f)

def update_prompts(candidate_details):
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
    agent = candidate_details.get('agent', {})
    
    # Set default values if agent isn't available
    job_title = agent.get('title', 'the position') if agent else 'the position'
    company_values = agent.get('company_values', '') if agent else ''
    job_description = agent.get('description', '') if agent else ''
    personality_type = agent.get('personality', {}).get('type', 'professional') if agent else 'professional'
    criteria = agent.get('criteria', []) if agent else []
    
    # Personalize the system prompt based on agent information
    personality_style = ""
    if personality_type == "friendly":
        personality_style = "Be friendly, warm and approachable. Use casual language while remaining professional."
    elif personality_type == "technical":
        personality_style = "Be technically focused and detailed. Ask in-depth questions about technical skills."
    elif personality_type == "professional":
        personality_style = "Maintain a formal and professional demeanor throughout the interview."
    elif personality_type == "custom" and agent and agent.get('personality', {}).get('description'):
        personality_style = agent.get('personality', {}).get('description')
    
    # Construct criteria text if available
    criteria_text = ""
    if criteria:
        criteria_text = "Pay special attention to these criteria in the candidate's background:\n"
        criteria_text += "\n".join([f"- {criterion}" for criterion in criteria])
    
    system_prompt = f"""Your name is Ari and you are conducting a behavioural interview for {candidate_name} for the role of {job_title}.
    This should be your personality style: {personality_style}
    
    This is some of the candidate criteria: {criteria_text}
    
    Company values: {company_values}
    
    Job description: {job_description}
    
    Initially start the interview with introductions, and small questions for a little bit.
    You have access to the candidate's resume with the function query_knowledge_database.
    Don't mention the name of this function to the candidate.
    You can use the function to ask any question about his/her resume, which you should a couple times in the interview, up to your discretion.
    You can also query it if a follow-up question regarding the resume might be helpful at the point in the interview.
    Your output will be converted to audio so don't include special characters in your answers.
    """
    
    return RAG_PROMPT, system_prompt

def get_query_knowledge_base_function(RAG_PROMPT):
    # This closure captures RAG_PROMPT so that the registered function has access to it.
    async def query_knowledge_base(function_name, tool_call_id, arguments, llm, context, result_callback):
        logger.info(f"Querying knowledge base for question: {arguments['question']}")
        client = genai.GenerativeModel(
            model_name="gemini-2.0-flash-lite-preview-02-05",
            system_instruction=RAG_PROMPT,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=64,
            ),
        )
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
    return query_knowledge_base

async def initialize_room(session):
    tavus = TavusVideoService(
        api_key=load_api_keys("tavus_api_key"),
        replica_id=load_api_keys("tavus_replica_id"),
        session=session,
        sample_rate=16000,
    )
    persona_name = await tavus.get_persona_name()
    room_url = await tavus.initialize()
    logger.info(f"Room URL is {room_url} and will be emailed to the candidate.")
    return tavus, persona_name, room_url

def send_interview_email(candidate_details, room_url):
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
    smtp_server = 'smtp.gmail.com'
    port = 465
    sender_email = load_api_keys("email")
    sender_password = load_api_keys("email_password")
    
    send_email(smtp_server, port, sender_email, sender_password, recipient_email, subject, body)
    logger.info(f"Interview email sent to {recipient_email}")

async def run_pipeline(session, system_prompt, room_url, tavus, persona_name, RAG_PROMPT):
    stt = DeepgramSTTService(api_key=load_api_keys("deepgram_api_key"))
    tts = CartesiaTTSService(
        api_key=load_api_keys("cartesia_api_key"),
        voice_id=load_api_keys("cartesia_voice_id"),
    )
    llm = GoogleLLMService(
        api_key=load_api_keys("google_gemini_api_key"),
        model="gemini-2.0-flash-001"
    )
    
    llm.register_function("query_knowledge_base", get_query_knowledge_base_function(RAG_PROMPT))
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

    messages = [{"role": "system", "content": system_prompt}]
    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

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

async def process_interview_request():
    candidate_details = load_candidate_details()
    
    # Configure Google Gemini API
    genai.configure(api_key=load_api_keys("google_gemini_api_key"))
    
    async with aiohttp.ClientSession() as session:
        try:
            logger.info("Initializing interview room...")
            tavus, persona_name, room_url = await initialize_room(session)
            
            logger.info(f"Sending interview link to candidate: {candidate_details.get('name', 'Candidate')}")
            send_interview_email(candidate_details, room_url)
            
            logger.info("Updating prompts with candidate details...")
            RAG_PROMPT, system_prompt = update_prompts(candidate_details)
            
            logger.info("Starting interview pipeline...")
            await run_pipeline(session, system_prompt, room_url, tavus, persona_name, RAG_PROMPT)
            
            return True, "Interview process completed successfully"
        except Exception as e:
            logger.error(f"Error in interview process: {str(e)}")
            return False, str(e)

def start_interview_process():
    try:
        # Directly run the async process in the main thread.
        success, message = asyncio.run(process_interview_request())
        return success, message
    except Exception as e:
        logger.error(f"Error in interview process: {e}")
        return False, str(e)

if __name__ == "__main__":
    success, message = start_interview_process()
    print(f"Process {'succeeded' if success else 'failed'}: {message}")
