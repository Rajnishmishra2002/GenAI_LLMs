from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

# # Calling the api and verifying that it's getted loaded


# def api_token():

#     load_dotenv()
#     api_key = os.get("HUGGINGFACEHUB_API_TOKEN")

#     if not api_token:
#         raise ValueError("NO API TOKEN FOUND ")

#     return api_key

# Defining the msg prompt


def msg():
    chat_template = ChatPromptTemplate([
        ('system', 'you are a helpful customer support agent'),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{query}')
    ])
    return chat_template

# Loading the chat history


def load_history():
    chat_history = []
    with open('chat_history.txt') as f:
        chat_history.extend(f.readlines())

    return chat_history

# calling the model


def main():

    # api_key = api_token()
    chat_template = msg()
    history = load_history()

    prompt = chat_template.invoke(
        {'chat_history': history, 'query': 'Where is my refund'})

    print(prompt)


if __name__ == "__main__":
    main()
