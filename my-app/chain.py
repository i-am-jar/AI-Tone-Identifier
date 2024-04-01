"""This is a template for a custom chain.

Edit this file to implement your chain logic.
"""

from langchain.chat_models.openai import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.runnable import Runnable

tone_func = {
    "name": "tone_identifier",
    "description": "Identifies the tone of a given writing sample",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The writing sample to analyze for tone"
            }
        },
        "required": ["text"]
    }
}


def get_chain() -> Runnable:
    """Return a chain."""
    prompt = ChatPromptTemplate.from_template("Identify the tone of the following writing sample: {topic}")
    model = ChatOpenAI().bind(functions=[tone_func], function_call={"name": "tone_identifier"})
    parser = JsonOutputFunctionsParser()
    return prompt | model | parser