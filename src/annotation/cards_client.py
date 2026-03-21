"""
Handles communication with the CARDS API.
Includes retry logic and returns structured output.
"""

from openai import OpenAI, APIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from taxonomy import Categories


# Initialize API client
client = OpenAI(
    api_key="dl-cards-d45b2a74558062a3effa9be37258b85f2dd07c6df246549c",
    base_url="https://api.discourselab.ai/v1"
)


@retry(
    retry=retry_if_exception_type(APIError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True
)
def classify_text(text: str):
    """
    Sends a single text to the CARDS API and returns:
    - structured category predictions
    - token usage information
    """

    response = client.beta.chat.completions.parse(
        model="cards-mini-sonnet-2024-12-05",
        messages=[{"role": "user", "content": text}],
        response_format=Categories,
        extra_body={"prompt_id": "cards"},
        temperature=0
    )

    parsed_output = response.choices[0].message.parsed

    return {
        "categories": parsed_output.categories,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens
    }