load_dotenv()

# %pip install langchain_google_genai load_dotenv crewai crewai_tools langchain_community langchain sentence-transformers langchain-groq langchain_huggingface --quiet openai gradio huggingface_hub

import os
import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 

from crewai import Agent, Task, Crew

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
load_dotenv()

from crewai import Agent, Task, Crew
from crewai import LLM
os.environ['GEMINI_API_KEY'] = os.environ.get('GEMINI_API_KEY')
llm = LLM(model="gemini/gemini-1.5-flash")
os.environ['GROQ_API_KEY'] = os.environ.get('GROQ_API_KEY')

"""TOOL"""

import crewai_tools
from crewai import tools

# from crewai_tools import ScrapeWebsiteTool

# docs_scrape_tool = ScrapeWebsiteTool(
#     website_url= {url}
# )

from crewai_tools import SerperDevTool
tool = SerperDevTool()

"""AGENTS"""

scrapper_agent = Agent(
    role="Senior Scrapper Representative",
	goal="Be the most friendly and helpful "
        "Scrapper representative in your team to scrape information inputted by user of query {query} ",
	backstory=(
		"You have scrapped many information inputted by user of query {query} and "
        "you are good and perfect at it and makes this task easy "
		"You need to make sure that you provide the best support!"
		"Make sure to provide full complete answers, "
        " and make no assumptions."
	),
	allow_delegation=False,
	llm=llm,
	verbose=True
)

Provider_agent = Agent(
    role="Senior information Provider Representative",
	goal="Be the most friendly and helpful "
        "information provider in your team to provide the information scrapped from web browser",
	backstory=(
		"You have provided many information that were scrapped by other agent from web browser and "
        "you are good and perfect at it and makes this task easy "
		"You need to make sure that you provide the best support!"
		"Make sure to provide full complete answers, "
        " and make no assumptions."
	),
	allow_delegation=False,
	llm=llm,
	verbose=True
)

"""TASK"""

scrapper_task = Task(
    description=(
        "user just reached out with a super important task"
	    "to scrape information from web browser of query {query} "
		"Make sure to use everything you know "
        "to provide the best support possible."
		"You must strive to provide a complete "
        "and accurate response to the user's query."
    ),
    expected_output=(
	    "A detailed, informative response to the "
        "user's query that addresses "
        "all aspects of their question.\n"
        "The response should include references "
        "to everything you used to find the answer, "
        "including external data or solutions. "
        "Ensure the answer is complete, "
		"leaving no questions unanswered, and maintain a helpful and friendly "
		"tone throughout."
    ),
	tools=[tool],
    agent=scrapper_agent,
)

Provider_task = Task(
    description=(
        "Your task is to make proper documented information that are scrapped from other agent "
		"Make sure to use everything you know "
        "to provide the best support possible."
		"You must strive to provide a complete "
        "and accurate response to the user's query."
    ),
    expected_output=(
	    "A detailed, informative response to the "
        "user's query that addresses and make it well and perfect dcumented to easily readable "
        "all aspects of their question.\n"
        "The response should include references "
        "to everything you used to find the answer, "
        "including external data or solutions. "
        "Ensure the answer is complete, "
		"leaving no questions unanswered, and maintain a helpful and friendly "
		"tone throughout."
    ),
    agent=Provider_agent,
)

crew = Crew(
  agents=[scrapper_agent, Provider_agent],
  tasks=[scrapper_task, Provider_task],
  verbose=True
)

# inputs = {
#     "query": input("Enter your query: "),
#     # "url": input("Enter which source to use for query: ")
# }
# result = crew.kickoff(inputs=inputs)
def get_text_response(message, history):
    result = crew.kickoff(inputs={"query": message})
    return result.raw

# from IPython.display import Markdown
# Markdown(result.raw)

"""Gradio - To create interface of chatbot"""

demo = gr.ChatInterface(get_text_response, examples=["How are you doing?","What are your interests?","Which places do you like to visit?"])

if __name__ == "__main__":
    demo.launch() 

