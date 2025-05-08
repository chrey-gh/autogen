import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

from dotenv import load_dotenv
import os   

load_dotenv()

async def main():
    # Create an OpenAI model client.
    model_client = AzureOpenAIChatCompletionClient(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        model=os.getenv("AZURE_OPENAI_MODEL_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        #azure_ad_token_provider=token_provider,  # Optional if you choose key-based authentication.
        api_key=os.getenv("AZURE_API_KEY") # For key-based authentication.
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

    await team.reset()  # Reset the team for a new task.
    await Console(team.run_stream(task="Write a short poem about the fall season."))  # Stream the messages to the console.

    # Use `asyncio.run(...)` when running in a script.
    #result = await team.run(task="Write a short poem about the fall season.")
    #print(result)
    # Close the model client connection.
    await model_client.close()


asyncio.run(main())

# 
# # Note: The `asyncio.run(main())` is used to run the main function in an asyncio event loop.
# # In a real-world application, you would typically have more complex error handling and cleanup.