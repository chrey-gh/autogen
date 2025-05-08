import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient



async def main():
    # Create an OpenAI model client.
    model_client = AzureOpenAIChatCompletionClient(
        azure_deployment="gpt-4o",
        model="gpt-4o-2024-11-20",
        api_version="2025-01-01-preview",
        azure_endpoint="https://westus.api.cognitive.microsoft.com/",
        #azure_ad_token_provider=token_provider,  # Optional if you choose key-based authentication.
        api_key="06fa986825f54cd0a4cb4fe4669a8db3", # For key-based authentication.
    )

    # Create the primary agent.
    primary_agent = AssistantAgent(
        "primary",
        model_client=model_client,
        system_message="You are a helpful AI assistant.",
    )

    # Create the critic agent.
    critic_agent = AssistantAgent(
        "critic",
        model_client=model_client,
        system_message="Provide constructive feedback. Respond with 'APPROVE' to when your feedbacks are addressed.",
    )

    # Define a termination condition that stops the task if the critic approves.
    text_termination = TextMentionTermination("APPROVE")

    # Create a team with the primary and critic agents.
    team = RoundRobinGroupChat([primary_agent, critic_agent], termination_condition=text_termination)

    # Use `asyncio.run(...)` when running in a script.
    result = await team.run(task="Write a short poem about the fall season.")
    print(result)
    # Close the model client connection.
    await model_client.close()


asyncio.run(main())

# 
# # Note: The `asyncio.run(main())` is used to run the main function in an asyncio event loop.
# # In a real-world application, you would typically have more complex error handling and cleanup.