from autogen_ext.auth.azure import AzureTokenProvider
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential
from autogen_agentchat.messages import UserMessage

import asyncio
import logging
from autogen_core import EVENT_LOGGER_NAME

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
    azure_deployment="gpt-4o",
    model="gpt-4o-2024-11-20",
    api_version="2025-01-01-preview",
    azure_endpoint="https://westus.api.cognitive.microsoft.com/",
    #azure_ad_token_provider=token_provider,  # Optional if you choose key-based authentication.
    api_key="06fa986825f54cd0a4cb4fe4669a8db3", # For key-based authentication.
)

async def main():
    result = await az_model_client.create([UserMessage(content="What is the capital of France?", source="user")])
    print(result)
    await az_model_client.close()

asyncio.run(main())