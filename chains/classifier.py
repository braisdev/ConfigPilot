from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal, Any


class Classifier(BaseModel):
    """
    This is a classifier that classify if the user input is related with the json fields provided
    or not.

    Attributes:
        related_with_fields: If the user input is related with the json fields provided or not
    """
    related_with_fields: Literal[True, False] = Field(
        description="If the user input is related with the json fields provided or not"
    )


def classify_input(user_input) -> Any:

    print("-- CLASSIFY INPUT --")

    system_message = SystemMessage(
        """You are an expert at determining whether a user's input provides actual attribute values corresponding to the 
        provided anime-character JSON fields.
        
        Your task is to analyze the user's input and decide if it includes specific information that can be directly 
        mapped to any of the anime-character fields. Focus exclusively on inputs that provide attribute values regarding
        only about the anime-character, and disregard questions, requests for guidance, or unrelated content.
    
        The anime-character JSON fields are:
        
        - name
        - age
        - gender
        - physical_appearance
        - personality
        - abilities_power
        - occupation
        
        Please provide a simple, clear determination: respond with `True` if the user's input includes actual attribute 
        values for any of the JSON fields, or `False` if it does not.
        
        **Examples:**
        
        - **Input:** "The character is a 16-year-old girl with long blue hair."
        
          - **Response:** `True`
        
        - **Input:** "Should I describe the character's abilities?"
        
          - **Response:** `False`
        
        - **Input:** "He is a wise old man who controls time."
        
          - **Response:** `True`
        
        - **Input:** "I need help coming up with a name."
        
          - **Response:** `False`
            """
        )

    human_message = HumanMessage(
        f"""Please assess the following user input and determine if it provides actual attribute values that correspond 
        to the provided JSON fields for an anime character profile.
    
        User Input: {user_input}
    
        Based on the specified JSON fields, does the user input include attribute values for any of them? Please answer 
        `True` or `False`."""
    )

    prompt_components = [
        system_message,
        human_message,
    ]

    prompt_template = ChatPromptTemplate.from_messages(prompt_components)

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0).with_structured_output(
        Classifier, method="json_schema", strict=True)

    classifier_chain = prompt_template | llm

    classification = classifier_chain.invoke({"user_input": user_input})

    return classification


if __name__ == "__main__":

    user_query = "Me llamo Brais"

    result = classify_input(user_query)

    print(result)
