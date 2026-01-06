from typing import Dict
from langchain_core.prompts import ChatPromptTemplate

graph_prompt_templates: Dict[str, Dict[str, str]] = {
    "bar": {
        "system": '''You are a data visualization expert. Given a question and some data, provide a concise and relevant structure for a bar chart.''',
        "human": '''Question: {question}
        Data: {data}

        Provide a bar chart structure in the following valid JSON format:
        {{
        "labels": ["string"],
        "values": [{{ "data": [0], "label": "string" }}]
        }}
        Ensure the structure is relevant to the question and data provided. Return ONLY the JSON object.'''
    },
    "horizontal_bar": {
        "system": '''You are a data visualization expert. Given a question and some data, provide a concise and relevant structure for a horizontal bar chart.''',
        "human": '''Question: {question}
        Data: {data}

        Provide a horizontal bar chart structure in the following valid JSON format:
        {{
        "labels": ["string"],
        "values": [{{ "data": [0], "label": "string" }}]
        }}
        Ensure the structure is relevant to the question and data provided. Return ONLY the JSON object.'''
    },
    "line": {
        "system": '''You are a data visualization expert. Given a question and some data, provide a concise and relevant structure for a line graph.''',
        "human": '''Question: {question}
        Data: {data}

        Provide a line graph structure in the following valid JSON format:
        {{
        "xValues": ["string"],
        "yValues": [{{ "data": [0], "label": "string" }}]
        }}
        Ensure the structure is relevant to the question and data provided. Return ONLY the JSON object.'''
    },
    "pie": {
        "system": '''You are a data visualization expert. Given a question and some data, provide a concise and relevant structure for a pie chart.''',
        "human": '''Question: {question}
        Data: {data}

        Provide a pie chart structure in the following valid JSON format:
        [
        {{ "label": "string", "value": 0 }}
        ]
        Ensure the structure is relevant to the question and data provided. Return ONLY the JSON array.'''
    },
    "scatter": {
        "system": '''You are a data visualization expert. Given a question and some data, provide a concise and relevant structure for a scatter plot.''',
        "human": '''Question: {question}
        Data: {data}

        Provide a scatter plot structure in the following valid JSON format:
        {{
        "series": [
            {{
                "data": [{{ "x": 0, "y": 0, "id": 0 }}],
                "label": "string"
            }}
        ]
        }}
        Ensure the structure is relevant to the question and data provided. Return ONLY the JSON object.'''
    }
}


def get_prompt(graph_type: str) -> Dict[str, str]:
    if graph_type not in graph_prompt_templates:
        raise ValueError(f"Unknown graph type: {graph_type}")

    template = graph_prompt_templates[graph_type]
    return ChatPromptTemplate.from_messages([
        ("system", template["system"]),
        ("human", template["human"]),
    ])
