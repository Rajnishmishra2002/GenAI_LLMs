from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
from langchain_community.llms import huggingface_endpoint


def load_api_token():
    # Load Variables from .env
    load_dotenv()

    # Get the token from environment
    api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

    # To check if it loaded correctly
    if not api_token:
        raise ValueError("API token not found")

    return api_token


def initialize_model():
    # Initialize the HuggingFacePipeline with the desired model and parameters
    llm = HuggingFacePipeline.from_model_id(
        model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        task='text-generation',
        pipeline_kwargs={
            'temperature': 0.7,  # Adjusted temperature for more diverse responses
            'max_new_tokens': 150  # Increased token limit for longer responses
        }
    )

    return ChatHuggingFace(llm=llm)


def create_msg():
    # Defining the msg Prompt
    return [
        SystemMessage(content="You are a helpful assistant "),
        HumanMessage(content="Tell me about diffrence between library and framework ")]


def main():
    load_api_token()
    chat_model = initialize_model()
    message = create_msg()
    # Calling the model Using .invoke
    result = chat_model.invoke(message)
    message.append(AIMessage(content=result.content))
    print(message)


if __name__ == "__main__":
    main()
