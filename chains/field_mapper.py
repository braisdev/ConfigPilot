from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from state import AnimeCharacter


def map_fields(user_input: str) -> AnimeCharacter:
    """
    Given a user input that has already been classified as having
    relevant anime character attribute data, parse and map it into
    the AnimeCharacter model fields.
    """

    system_message = SystemMessage(
        """You are a highly accurate information extraction assistant. Given the user input describing an anime character,
        extract the character's attributes and map them into the following JSON fields:

        - name (string)
        - age (integer)
        - gender (string)
        - physical_appearance (string)
        - personality (string)
        - abilities_power (string)
        - occupation (string)

        Instructions:
        - If an attribute is not mentioned, leave it as null (None).
        - If an attribute is implicitly described, interpret it reasonably.
        - For example, if the user says "He is a 3-year-old boy," then:
            name: None (not mentioned)
            age: 3
            gender: "male"
            physical_appearance: None (not enough info)
            personality: None (not mentioned)
            abilities_power: None (not mentioned)
            occupation: None (not mentioned)

        Return only the final JSON object as structured output.
        """
    )

    human_message = HumanMessage(
        f"""User input: {user_input}

Please extract the fields and return them in JSON format."""
    )

    prompt_template = ChatPromptTemplate.from_messages([system_message, human_message])

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0).with_structured_output(
        AnimeCharacter, method="json_schema", strict=True
    )

    extraction_chain = prompt_template | llm

    extracted_character = extraction_chain.invoke({"user_input": user_input})
    return extracted_character


if __name__ == "__main__":
    # Example usage:
    user_query = "He is a 3-year-old boy"
    # This should map age=3, gender="male" (or "boy"), occupation="temple guard", physical_appearance="wears a red
    # kimono", etc.
    character_data = map_fields(user_query)
    print(character_data)
