

import warnings
warnings.filterwarnings("ignore")

# Import necessary libraries
import os
import autogen
import gradio as gr
from openai import OpenAI  # Keep for reference if needed
from dotenv import load_dotenv
from IPython.display import display, Markdown
import random  # Used later for unique Gradio outputs
import google.generativeai as genai  # Import the Google library

# Load environment variables from the .env file
load_dotenv()

# Retrieve API keys from environment variables
openai_api_key = "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"
google_api_key = os.getenv("GOOGLE_API_KEY")  # Load the Google API key

print("Setup Complete: Libraries installed and API keys loaded.")
config_list_openai = [
    {
        "model": "meta/llama-3.1-8b-instruct",
        "api_key": openai_api_key,
        "base_url": "https://integrate.api.nvidia.com/v1"
    }
]

llm_config_openai = {
    "config_list": config_list_openai,
    "temperature": 0.7,  # Use a slightly higher temp for creative marketing ideas
    "timeout": 120,
}
message = "We need to design a cron job to update / insert 5Lacs record each day."
architect_agent = autogen.ConversableAgent(name="AWS_services_Solution_architect", system_message="Need to design long running cron job suggest ideas", llm_config=llm_config_openai, human_input_mode="NEVER")
devloper_agent = autogen.ConversableAgent(name="AWS_python_developer", system_message="Need to design long running cron job suggest ideas. code hanges in short ", llm_config=llm_config_openai, human_input_mode="NEVER")

chat_result_openai_only = architect_agent.initiate_chat(recipient=devloper_agent, max_turns=4, message=message)
print_chat_history(chat_result_openai_only)
chat_result_openai_only = architect_agent.initiate_chat(recipient=devloper_agent, max_turns=4, message=message)
print_chat_history(chat_result_openai_only)
