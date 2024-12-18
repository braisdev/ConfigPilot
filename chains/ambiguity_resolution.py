from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def generate_response(input_type: str, reasoning: str) -> Any:
    system_message = SystemMessage("""
            You are a helpful assistant that generates specific clarification questions based on the analysis of user input. 
            Given the input type (ambiguous, misleading, or stable) and the reasoning behind this classification, your task is to 
            generate a concrete question aimed at clarifying the ambiguous or misleading field(s). 
            If multiple fields are identified as ambiguous or misleading, ask about one field at a time, prioritizing the most critical one.
        """)

    human_message = HumanMessage(f"""
            Based on the previous analysis, the input has been classified as '{input_type}'. The reasoning provided is:

            "{reasoning}"

            Please generate a concrete question to clarify the ambiguous or misleading field in the user's input.
        """)

    prompt_components = [system_message, human_message]

    prompt_template = ChatPromptTemplate.from_messages(prompt_components)

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5)

    relevance_reflector_chain = prompt_template | llm

    relevance = relevance_reflector_chain.invoke({"input_type": input_type,
                                                  "reasoning": reasoning})

    return relevance


if __name__ == "__main__":
    input_type = "ambiguous"
    reasoning = "The input '15 años' indicates an age but does not specify the name, gender, or any other relevant fields. It lacks clarity and detail about the individual, making it open to multiple interpretations."
    response = generate_response(input_type=input_type, reasoning=reasoning)
    print(response.content)


