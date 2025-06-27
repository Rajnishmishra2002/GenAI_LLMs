# LangChain Learning Project 🚀

Welcome to my LangChain learning journey! In this project, I’m exploring how to build powerful applications by leveraging **LangChain**, a framework designed to simplify the use of large language models (LLMs) in various workflows.


## Project Topics

###  **Prompt Engineering**

* Learning how to design effective prompts for LLMs.
* Experimenting with different styles of prompts to extract better responses.

###  **Chains & Workflows**

* Building sequences of LLM interactions using **LangChain Chains**.
* Integrating different modules (prompt templates, models, parsers) to create complex workflows.

### 🔗 **Parallel Chains & Merging**

* Running parallel tasks (like summarization and quiz generation) and merging their results for final output.
* Using **RunnableParallel** and **prompt merging** to create streamlined pipelines.

---

## 🤖 Models Used

* **HuggingFace Models**: Working with various open-source models like TinyLlama, T5, etc.
* **HuggingFacePipeline**: Creating pipelines for tasks like summarization, text generation, and more.

---

##  Technologies & Libraries

* **LangChain**: For building prompt templates, chains, and managing LLM workflows.
* **HuggingFace Transformers**: Model inference and pipeline creation.
* **Python**: Scripting and automation.
* **Flask** (planned): For web integration and exposing APIs (future enhancement).

---

##  Key Features

 Create effective **prompt templates** for LLM tasks.
 Run **parallel tasks** (e.g., summarization and quiz generation).
 Merge outputs from different tasks using **prompt merging**.
 Learn about **model limitations** (e.g., not all models support all tasks!).
 Use environment variables for secure API key handling (via `.env` and `load_dotenv`).

---

## 🔍 Example Workflow

```python
# 1 Load environment variables
load_dotenv()

# 2 Create prompt templates for notes and quizzes
prompt1 = PromptTemplate(template="Give notes from the given {text}", input_variables=['text'])
prompt2 = PromptTemplate(template="Create a quiz from the given {text}", input_variables=['text'])

# 3 Run tasks in parallel and merge their outputs
parallel_chains = RunnableParallel({
    'notes': prompt1 | model | parser,
    'quiz': prompt2 | model | parser
})

merge_chain = prompt3 | model | parser

# 4 Invoke the final merged output
result = merge_chain.invoke({'notes': '...', 'quiz': '...'})
```

---

##  Future Improvements

*  Incorporate **memory** and **retrieval** for more context-aware outputs.
*  Build a Flask API to expose LangChain-powered endpoints.
*  Add data visualization and insights using **Power BI** or **Streamlit**.


##  Notes & Troubleshooting

* Some models do not support all tasks (e.g., summarization not supported for LLaMa models).
* When using **PromptTemplates**, ensure input variables match the expected keys.


##  License

This project is for **educational purposes** only.


##  Acknowledgements

Thanks to the **LangChain** and **HuggingFace** teams for providing these powerful tools!

