from autogen_ext.auth.azure import AzureTokenProvider
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential
from autogen_agentchat.messages import UserMessage

import asyncio
import logging
from autogen_core import EVENT_LOGGER_NAME

import os
from dotenv import load_dotenv  

load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


# Create the token provider
# token_provider = AzureTokenProvider(
#     DefaultAzureCredential(),
#     "https://cognitiveservices.azure.com/.default",
# )

az_model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    model=os.getenv("AZURE_OPENAI_MODEL_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    #azure_ad_token_provider=token_provider,  # Optional if you choose key-based authentication.
    api_key=os.getenv("AZURE_API_KEY") # For key-based authentication.
)

async def main():
    result = await az_model_client.create([UserMessage(content="What is the capital of France?", source="user")])
    print(result)
    await az_model_client.close()

asyncio.run(main())