from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class RelevanceReflector(BaseModel):
    """
    A utility class to assess and validate the consistency of user input.

    Attributes:
        input_type (Literal["ambiguous", "misleading", "stable"]):
            Indicates the nature of the user input:
                - "ambiguous": The input lacks clarity or is open to multiple interpretations.
                - "misleading": The input provides incorrect or deceptive information.
                - "stable": The input is consistent and provides sufficient information to continue populating
                JSON fields.
    """
    input_type: Literal["ambiguous", "misleading", "stable"] = Field(
        description="Indicates the nature of the user input, either ambiguous, misleading, or stable."
    )
    reasoning: str = Field(
        description="Explanation of the input type and its validity or not."
    )


def relevance_reflector(user_input: str):
    system_message = SystemMessage("""
        You are a validator responsible for analyzing user input to determine its consistency with a predefined set of JSON fields. 
        The JSON fields are:
        - name
        - age
        - gender
        - physical_appearance
        - personality
        - abilities_power
        - occupation

        Your task is to evaluate the user input and classify it into one of three categories:
        1. **Ambiguous**: Input does not mention any of the predefined fields or mentions fields but lacks clarity or sufficient detail for those fields.
        2. **Misleading**: Input contains contradictions or inaccuracies related to the fields it references.
        3. **Stable**: Input clearly mentions **at least one** of the predefined fields and provides sufficient detail for the mentioned field(s).

        **Important**:
        - **Only** consider the fields that the input **explicitly mentions**.
        - If the input mentions **one or more** fields and provides clear and sufficient information for **any** of those fields, classify it as **Stable**.
        - Do **not** consider the absence of other fields in your evaluation.
        - If the input does not mention any predefined fields or is unclear about the mentioned fields, classify it as **Ambiguous**.
        - If the input contains contradictions or inaccuracies within the mentioned fields, classify it as **Misleading**.

        **Do not**:
        - Assume any information about fields not mentioned.
        - Require that multiple fields be mentioned for classification as **Stable**.

        **Provide a brief explanation** (1-2 sentences) justifying your decision based solely on the provided information.

        **Examples**:

        **Example 1**:
        - **User Input**: "el personaje se llama brais"
        - **Classification**: Stable
        - **Explanation**: The input clearly provides the 'name' of the character as 'brais', fulfilling the required detail for the referenced field.

        **Example 2**:
        - **User Input**: "el personaje tiene super fuerza pero es muy tímido"
        - **Classification**: Stable
        - **Explanation**: The input mentions 'abilities_power' as 'super fuerza' and 'personality' as 'muy tímido', providing clear information for both fields.

        **Example 3**:
        - **User Input**: "el personaje es increíblemente fuerte pero no tiene ningún poder"
        - **Classification**: Misleading
        - **Explanation**: The input contains a contradiction regarding 'abilities_power'; being "increíblemente fuerte" suggests power, but it also states "no tiene ningún poder."

        **Example 4**:
        - **User Input**: "el personaje vive en un mundo fantástico"
        - **Classification**: Ambiguous
        - **Explanation**: The input does not mention any of the predefined fields.

        **Example 5**:
        - **User Input**: "el personaje se llama"
        - **Classification**: Ambiguous
        - **Explanation**: The input mentions the 'name' field but does not provide a name, lacking sufficient detail.
    """)

    human_message = HumanMessage(f"""
        Analyze the following user input and determine its consistency based on the predefined JSON fields.

        **User input:**
        {user_input}

        **Instructions:**
        - Classify the input as "ambiguous", "misleading", or "stable".
        - Provide a concise reasoning focusing only on the fields present in the input.
    """)

    prompt_components = [system_message, human_message]

    prompt_template = ChatPromptTemplate.from_messages(prompt_components)

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0).with_structured_output(
        RelevanceReflector, method="json_schema", strict=True)

    relevance_reflector_chain = prompt_template | llm

    relevance = relevance_reflector_chain.invoke({"user_input": user_input})

    return relevance


if __name__ == "__main__":
    user_query = "My name is Brais"
    relevance_output = relevance_reflector(user_query)
    print(relevance_output)

