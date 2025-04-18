## Design and Implementation of LangChain Expression Language (LCEL) Expressions

### AIM:
To design and implement a LangChain Expression Language (LCEL) expression that utilizes at least two prompt parameters and three key components (prompt, model, and output parser), and to evaluate its functionality by analyzing relevant examples of its application in real-world scenarios.

### PROBLEM STATEMENT:
Build a smart assistant using LangChain LCEL that provides information on smart watch and its features.

### DESIGN STEPS:
#### STEP 1:
Load product FAQs and convert them into vector embeddings using OpenAIEmbeddings.
#### STEP 2:
Store these embeddings in an in-memory vector store (DocArrayInMemorySearch) and create a retriever.
#### STEP 3:
Define a prompt template that takes context and question as input.
#### STEP 4:
Set up a language model (ChatOpenAI) and an output parser (StrOutputParser) to process and format responses.
#### STEP 5:
Build a chain using LCEL: it retrieves relevant context, formats the prompt, generates a response using the model, and parses the output.
#### STEP 6:
Invoke the chain with a user question and print the assistant’s answer.

### PROGRAM:
```py
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableMap
from langchain.schema.output_parser import StrOutputParser

faq_texts = [
    "The smartwatch battery lasts up to 7 days on a single charge.",
    "You can track your heart rate, sleep, and steps with the watch.",
    "The smartwatch is water-resistant up to 50 meters."
]
embedding = OpenAIEmbeddings()
vectorstore = DocArrayInMemorySearch.from_texts(faq_texts, embedding=embedding)
retriever = vectorstore.as_retriever()
template = """You are a helpful assistant answering questions based only on the following product FAQs:
{context}
Customer Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
output_parser = StrOutputParser()
chain = (
    RunnableMap({
        "context": lambda x: retriever.get_relevant_documents(x["question"]),
        "question": lambda x: x["question"]
    })| prompt| model| output_parser)

response = chain.invoke({"question": "Explain the features of the watch?"})
print(response)

```

### OUTPUT:
![output](./Screen%20Shot%201947-01-28%20at%2019.08.40.png)

### RESULT:
The LCEL expression successfully integrated two prompt parameters with prompt, model, and output parser components to generate personalized, context-aware responses. This demonstrates LangChain's ability to build dynamic, real-world applications like smart sales assistants.
