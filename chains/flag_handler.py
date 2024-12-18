from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from state import AnimeCharacter


def response_for_flags(user_input, extracted_character: AnimeCharacter, flags):

    system_message = SystemMessage(
        """
        You are a friendly and approachable assistant that helps users create anime characters.
        Your main job is to point out any flags—serious inconsistencies or errors—in the current character mapping provided by the user.

        When you spot a flag, you should:
        - Clearly explain what the issue is.
        - Ask specific questions to help the user clarify and resolve the problem.
        - Encourage the user to provide additional information or correct any mistakes to ensure the character mapping is accurate and consistent.

        Keep the tone casual and supportive.
        """
    )

    human_message = HumanMessage(
        f"""
        **User Input:** {user_input}

        **Current Field Mapping:**
        {extracted_character.model_dump_json(indent=2)}

        **Detected Flags:**
        {flags}

        **Request:** Please help me address the above flags. Provide guidance on how to clarify and fix these issues 
        to ensure the anime character is accurately and consistently mapped."""
    )

    prompt_components = [human_message, system_message]

    prompt = ChatPromptTemplate(prompt_components)

    llm = ChatOpenAI(model_name="gpt-4o")

    chain = prompt | llm

    response = chain.invoke({})

    return response


if __name__ == "__main__":
    # Example user input (in Spanish)
    user_query = "Mi personaje de anime se llama mi perro se llama Brais."

    # Extracted character fields (this would typically be done by a field extraction process)
    extracted_char = AnimeCharacter(
        name="Brais",
        age=None,
        gender=None,  # Not stated
        physical_appearance=None,
        personality=None,
        abilities_power=None,
        occupation=None  # Not mentioned
    )

    # Mocked ReflectionFeedback.flags
    mocked_flags = [
        "Name is not provided, there's should be a confusion."
    ]

    # Generate response based on mocked flags
    result = response_for_flags(user_query, extracted_char, mocked_flags)

    # Print the assistant's response
    print("Assistant's Response:")
    print(result.content)
