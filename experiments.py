# NOTE(tom): This is currently pasted from my Kaggle notebook where I do experiments.
# We first need to discuss this before integrating it.
from typing import Any, Union
import re

# -----------------------------------------------------------------
# LLM
# -----------------------------------------------------------------
from abc import ABC, abstractmethod

class LLM(ABC):
    """
    Abstract base class for language models.
    """

    def __init__(self, model_name: str):
        """
        Initialize the LLM with a model name.

        Args:
            model_name (str): The name of the model.
        """
        self.model_name = model_name



# -----------------------------------------------------------------
# AzureOpenAILLM
# -----------------------------------------------------------------
import os
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage

class AzureOpenAILLM(LLM):
    """
    Azure OpenAI implementation of the LLM class, supporting different GPT models.
    """

    def __init__(
        self,
        api_version: str,
        api_endpoint: str,
        api_key: str,
        model_version: str,
        temperature: float = 0.0, # 0.01
    ):
        """
        Initialize the AzureOpenAILLM with the specified API parameters, model version, and temperature.

        Args:
            api_version (str): The version of the Azure OpenAI API to use.
            api_endpoint (str): The Azure OpenAI endpoint.
            api_key (str): The API key for authentication with Azure OpenAI.
            model_version (str): The version of GPT to use (e.g., "gpt-3.5-turbo", "gpt-4").
            temperature (float, optional): The temperature setting for the model. Defaults to 0.0.
        """
        super().__init__(model_version)

        self.model = AzureChatOpenAI(
            openai_api_version=api_version,
            azure_endpoint=api_endpoint,
            openai_api_key=api_key,
            azure_deployment=model_version,
            temperature=temperature,
            verbose=True,
        )


DEFAULT_SUMMARY_TRIGGER_TOKEN = "PIXIE_SUMMARY_READY"

class QuestionAskerLLM:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.system_prompt_template = """
        You are a seasoned denim specialist at a high-end jeans boutique. 
        You are sophisticated, observant and your tone is effortlessly cool, but to the point.

        Objective: 
        Ask questions about the given questionnaire only.

        Topic:
        * denim, jeans, legware, clothing

        Domain Guardrail:
        * You are specialist your topic ONLY. 
        * If the user mentions other items or irrelevant topics, 
        acknowledge them briefly, but immediately pivot back to your topic.

        Questionnaire (Ask about the following, only if the fields are blank):
        {summary}
        
        General conversation rules:
        * Only ask about blank fields that still have a [] in it!
        * Ask one field at a time.
        * Only ask questions about the fields, do not come up with new fields!
        * Acknowledge & Elevate: briefly validate the last filled in field in the questionnaire
        * Visual Language: Use sensory words (e.g. texture, silhouette, drape, grit, washed-out, structured).
        * Language: Always respond in the language used in the questionnaire.

        Critical Termination Logic:
        - Look at the questionnaire. 
        - If there are no fields with still the empty list [] or if the user indicates they are finished, output {completion_token} and nothing else.

        New message:
        """

    def generate(self, summary: str = "", completion_token: str = DEFAULT_SUMMARY_TRIGGER_TOKEN) -> str | None:
        system_prompt = self.system_prompt_template.format(
            summary=summary,
            completion_token=completion_token
        )

        system_message = SystemMessage(content=system_prompt)
        try:
            response = self.llm.model.invoke([system_message])
            #if completion_token in response.content.strip(): 
            #    return None
            return response.content.strip()
        except Exception as e:
            print(f"Error: {str(e)}")
        return None


class QuestionnaireNotulistLLM:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.system_prompt_template = """
        Extract and update the user's denim preferences into a structured summary.
        If the user does not know a field, declines or gives a partial answer: still write it in.
        Formulate in a gender-neutral way.

        Topic:
        * denim, jeans, legware, clothing

        Domain Guardrail:
        * You are specialist on your topic ONLY.

        Fields to update (please translate these and the field names in the language of the current answer):
        0. Initial user question: []
        1. Occasion, Vibe, Setting: []
        2. What do you like to wear or what are your preferences: []
        3. Most Important Aspects: [] 
        4. Hardest thing when trying to find pants: []
        5. What size or fit do you have: []
        6. Color & Wash: []
        7. Fabric, material (e.g., denim or cotton, weight and tactile feel): []
        8. Length (e.g. full length, shorts, 7/8th): []

        *** Given information ***
        Current Summary: {summary}
        AI Question: {current_question}
        User Answer: {current_answer}

        If the answer contains information useful for other fields: fill it in here as well!

        Internally analyze in which language the answer is. 
        Use this language in the updated structured summary. 
        Updated Structured Summary (in the language of current answer):
        """

    def generate(self, current_question: str = "", current_answer: str = "", questionnaire: str = "") -> str | None:
        system_prompt = self.system_prompt_template.format(
            current_question=current_question,
            current_answer=current_answer,
            summary=questionnaire
        )
        system_message = SystemMessage(content=system_prompt)
        try:
            response = self.llm.model.invoke([system_message])
            return response.content
        except Exception as e:
            print(f"Error: {str(e)}")
        return None

class ConversationSummarizerLLM:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.system_prompt_template = """
        Given:
        * The current question of an AI
        * The answer of a user
        * The summary of previous information.
        Summarize everything what the user wants in 1 paragraph without losing any information.
        Formulate in a gender-neutral way.
        Internally analyze in which language all this information is. Use this language in the summary.
        
        AI current question:
        {current_question}
        User current answer:
        {current_answer}
        Summary of previous information:
        {summary}
        
        New summary of what user wants (translated to the language of all this information):
        """
    
    def generate(self, current_question: str = "", current_answer: str = "", summary: str = "") -> str | None:
        system_prompt = self.system_prompt_template.format(current_question=current_question, current_answer=current_answer, summary=summary)
        system_message = SystemMessage(content=system_prompt)
        try:
            response = self.llm.model.invoke([system_message])
            return response.content
        except Exception as e:
            print(f"Error: {str(e)}")
        return None


llm = AzureOpenAILLM(
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model_version=os.getenv("AZURE_OPENAI_MODEL_VERSION")
)
question_asker_llm = QuestionAskerLLM(llm=llm)
questionnaire_notulist_llm = QuestionnaireNotulistLLM(llm=llm)
conversation_summarizer_llm = ConversationSummarizerLLM(llm=llm)

summary: str = ""
questionnaire: str = ""
user_answer: str = ""

i = 0
while True:
    if i == 0:
        question = "Hoi! Ik help je graag de ideale jeans te vinden. Waar ben je vandaag naar op zoek?\n"
        question += "Hi! I’m here to help you find the perfect jeans. What are you looking for today?\n"
        question += "Salut! Je vous aide à trouver le jean idéal. Que cherchez-vous aujourd'hui?"
    else:
        question = question_asker_llm.generate(summary=questionnaire)
        if question is None:
            break
    
    question = "AI: " + question
    print(question)
    user_answer = input("User: ")
    user_answer = "User: " + user_answer
    if i == 0:
        question = "Initial question:"
    questionnaire = questionnaire_notulist_llm.generate(current_question=question, current_answer=user_answer, questionnaire=questionnaire)
    summary = conversation_summarizer_llm.generate(current_question=question, current_answer=user_answer, summary=summary) 
    print(questionnaire)
    print(summary)
    if not "[]" in questionnaire:
        break
    i += 1

print("Klaar!")
print(summary)
