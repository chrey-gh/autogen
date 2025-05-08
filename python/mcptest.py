from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

import asyncio
import os
from dotenv import load_dotenv  

load_dotenv()

# Get the fetch tool from mcp-server-fetch.
fetch_mcp_server = StdioServerParams(command="uvx", args=["mcp-server-fetch"])

async def main():
    # Create an MCP workbench which provides a session to the mcp server.
    async with McpWorkbench(fetch_mcp_server) as workbench:  # type: ignore
        # Create an agent that can use the fetch tool.
        model_client = AzureOpenAIChatCompletionClient(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            model=os.getenv("AZURE_OPENAI_MODEL_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            #azure_ad_token_provider=token_provider,  # Optional if you choose key-based authentication.
            api_key=os.getenv("AZURE_API_KEY") # For key-based authentication.
        )
        fetch_agent = AssistantAgent(
            name="fetcher", model_client=model_client, workbench=workbench, reflect_on_tool_use=True
        )

        # Let the agent fetch the content of a URL and summarize it.
        result = await fetch_agent.run(task="Summarize the content of https://en.wikipedia.org/wiki/Seattle")
        assert isinstance(result.messages[-1], TextMessage)
        print(result.messages[-1].content)

        # Close the connection to the model client.
        await model_client.close()

# Run the main function
asyncio.run(main())