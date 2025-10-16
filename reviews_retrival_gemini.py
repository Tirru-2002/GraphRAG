import os
import requests
from langchain_core.prompts import ChatPromptTemplate

# --- MOCK RETRIEVER ---
# This is a placeholder for your actual retriever function.
# It should take a question as input and return relevant reviews.
# For this example, it returns a fixed set of reviews.
def retriever(question: str) -> list[str]:
    """
    A mock retriever that returns predefined reviews based on keywords in the question.
    Replace this with your actual vector database retrieval logic.
    """
    print(f"Searching for reviews related to: '{question}'")
    if "deep dish" in question.lower():
        return [
            "The deep dish pizza was incredible, with a crispy crust and tons of cheese.",
            "I wasn't a fan of the deep dish, I found the sauce to be too sweet for my liking."
        ]
    elif "delivery" in question.lower():
        return [
            "Delivery was surprisingly fast, arriving 15 minutes earlier than quoted.",
            "The pizza arrived cold and the box was crushed. Very disappointing delivery experience.",
            "I've ordered delivery multiple times and it's always been reliable and on time."
        ]
    else:
        return [
            "This place has a great atmosphere and friendly staff.",
            "The classic pepperoni pizza is a must-try.",
            "Parking can be a bit of a challenge during peak hours."
        ]

# --- Gemini API Configuration ---
# It's recommended to get the API key from environment variables for security.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# The correct endpoint for Gemini Pro.
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"

def query_gemini_api(prompt: str):
    """
    Sends a prompt to the Gemini API and returns the generated text.
    """
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY environment variable not set."

    headers = {
        "Content-Type": "application/json"
    }
    # The payload structure for the Gemini API is different.
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }]
    }
    try:
        response = requests.post(GEMINI_API_URL, json=payload, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes (4xx or 5xx)
        # Correctly parse the response from the Gemini API
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except requests.exceptions.RequestException as e:
        return f"Error calling Gemini API: {e}"
    except (KeyError, IndexError) as e:
        return f"Error parsing Gemini API response: {response.json()}"


template = """
You are an expert in answering questions about a pizza restaurant.
You must summarize the provided reviews to answer the question. Do not make up information.

Here are some relevant reviews:
{reviews}

Based on these reviews, answer the following question:
Question: {question}
"""
prompt_template = ChatPromptTemplate.from_template(template)

def main():
    while True:
        print("\n\n-------------------------------")
        question = input("Ask your question (or type 'q' to quit): ")
        if question.lower() == "q":
            break
        
        # 1. Retrieve relevant reviews
        reviews = retriever(question)
        
        # 2. Format the prompt with the reviews and the question
        formatted_prompt = prompt_template.format(reviews="\n".join(f"- {r}" for r in reviews), question=question)
        
        print("\n--- Sending Prompt to Gemini ---")
        print(formatted_prompt)
        print("----------------------------------\n")
        
        # 3. Get the answer from the Gemini API
        result = query_gemini_api(formatted_prompt)
        print("Answer:")
        print(result)

if __name__ == "__main__":
    main()


## Explanation of the Code!

''' This code is a simple command-line application that allows users to ask questions about a pizza restaurant. It retrieves relevant reviews based on the user's question and uses the Gemini API to generate an answer based on those reviews. Here's a breakdown of the main components:

1. **Review Retrieval**: The `retriever` function is responsible for fetching relevant reviews from a data source. It uses the user's question to find reviews that may contain answers.

2. **Prompt Formatting**: The `ChatPromptTemplate` is used to create a structured prompt for the Gemini API. It includes the retrieved reviews and the user's question, ensuring that the API has all the necessary context to generate a relevant answer.

3. **API Interaction**: The `query_gemini_api` function handles communication with the Gemini API. It sends the formatted prompt and processes the API's response, returning the generated answer.

4. **User Interface**: The `main` function provides a simple command-line interface for users to interact with the application. It continuously prompts the user for questions, retrieves reviews, formats the prompt, and displays the API's response.

Overall, this code demonstrates how to integrate a language model API into a question-answering system, leveraging user input and external data (reviews) to provide informative responses.'''