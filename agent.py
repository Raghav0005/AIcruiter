import asyncio
import os
import sys
from typing import Any, Mapping
import ssl
import certifi

import aiohttp
from dotenv import load_dotenv
from loguru import logger

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


async def main():
    # Use the SSL context in the ClientSession
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
        tavus = TavusVideoService(
            api_key="878784f9f43442b69bbbff0865503bde",
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

        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your "
                "capabilities in a succinct way. Your output will be converted to audio so don't "
                "include special characters in your answers. Respond to what the user said in a "
                "creative and helpful way.",
            },
        ]

        context = OpenAILLMContext(messages)
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