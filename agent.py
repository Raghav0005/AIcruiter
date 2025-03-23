import asyncio
import json
import os
import sys
import time
from typing import Any, Mapping
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

RAG_MODEL = "gemini-2.0-flash-lite-preview-02-05"
RAG_CONTENT = get_rag_content()
genai.configure(api_key="AIzaSyAQRHz9zd9JX0PXx80TxAO8oBmFvarYVyo")

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
{RAG_CONTENT}
"""

async def query_knowledge_base(
    function_name, tool_call_id, arguments, llm, context, result_callback
):
    """Query the knowledge base for the answer to the question."""
    logger.info(f"Querying knowledge base for question: {arguments['question']}")
    client = genai.GenerativeModel(
        model_name=RAG_MODEL,
        system_instruction=RAG_PROMPT,
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=64,
        ),
    )
    # for our case, the first two messages are the instructions and the user message
    # so we remove them.
    conversation_turns = context.messages[2:]
    # convert to standard messages
    messages = []
    for turn in conversation_turns:
        messages.extend(context.to_standard_messages(turn))

    def _is_tool_call(turn):
        if turn.get("role", None) == "tool":
            return True
        if turn.get("tool_calls", None):
            return True
        return False

    # filter out tool calls
    messages = [turn for turn in messages if not _is_tool_call(turn)]
    # use the last 3 turns as the conversation history/context
    messages = messages[-3:]
    messages_json = json.dumps(messages, ensure_ascii=False, indent=2)

    logger.info(f"Conversation turns: {messages_json}")

    start = time.perf_counter()
    response = client.generate_content(
        contents=[messages_json],
    )
    end = time.perf_counter()
    logger.info(f"Time taken: {end - start:.2f} seconds")
    logger.info(response.text)
    await result_callback(response.text)

async def main():
    # Use the SSL context in the ClientSession
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
        tavus = TavusVideoService(
            api_key="2a160544227e45e4b26b5b1418455a48",
            replica_id="raff7843cc3d",
            session=session,
            sample_rate=16000,
        )

        persona_name = await tavus.get_persona_name()
        room_url = await tavus.initialize()

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
        tools = [
            {
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
            },
        ]

        system_prompt = """You are conducting a behavioural interview for Raghav Vasudeva on behalf of Google.
                            Initially start the interview with introductions, and small questions for a little bit.
                            You have access to his resume with the function query_knowledge_database. You can use the function
                            to ask any question about his/her resume, and you should a couple times in the interview, up to your discretion.
                            You can also query it if a follow-up question regarding the resume might be helpful at the point in the interview.
                            Your output will be converted to audio so don't 
                            include special characters in your answers (like .js, or extensions, etc.) Respond to what the user said in a
                            creative and helpful way.
                            """
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
        ]

        context = OpenAILLMContext(messages, tools)
        context_aggregator = llm.create_context_aggregator(context)
        
        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                context_aggregator.user(),
                llm,
                tts,
                context_aggregator.assistant(),
                tavus,
                transport.output(),
            ]
        )

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
        async def on_participant_joined(
            transport: DailyTransport, participant: Mapping[str, Any]
        ) -> None:
            logger.debug(f"Ignoring {participant['id']}'s microphone")
            await transport.update_subscriptions(
                participant_settings={
                    participant["id"]: {
                        "media": {"microphone": "unsubscribed"},
                    }
                })
            await task.queue_frames(
                [context_aggregator.user().get_context_frame()]
            )
            

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            await task.cancel()

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())