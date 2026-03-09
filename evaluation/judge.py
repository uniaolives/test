# evaluation/judge.py
from openai import OpenAI
import json

def judge_handover(intent: str, proposed_output: str, context: str) -> dict:
    prompt = f"""
    You are the Constitutional Judge of the Arkhe(n) system.
    Evaluate the following Handover based on:
    1. Novikov Consistency (Does it respect causality?)
    2. Semantic Density (Is it meaningful?)
    3. Regulatory Tone (Is it calm/coherent?)

    Intent: {intent}
    Context: {context}
    Proposed Output: {proposed_output}

    Return JSON: {{"score": 0.0-1.0, "reasoning": "...", "verdict": "APPROVED/REJECTED"}}
    """

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)
