import requests
from transformers import pipeline
import json
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device='cuda')

url = "http://localhost:11434/api/generate"
data = {
    "model": "llama3.1-8B-Instruct",
    "prompt": "",
    "stream": False
}
def generate(query, context):
    with open("history.json", "r") as json_file:
                hist = json.load(json_file)
    prompt = f"""You are a helpful, knowledgeable virtual assistant for Swisscom AG, a leading Swiss telecommunications company. Your goal is to assist users with their inquiries efficiently, providing accurate information and excellent customer service. Maintain a polite and professional tone throughout the conversation. If a user has complex or sensitive issues that require human assistance, provide clear guidance on how they can get further help.” Prompt Components:
Greeting and Context Recognition Detect language preference: “Hello! How can I help you today?” (e.g., adapt to German, French, Italian, English based on user profile or language input).
Understanding User Needs and Intent Example questions to clarify user intent:
“Could you please provide more details about [specific issue, e.g., mobile plan, internet connection]?”
“Are you interested in information about [e.g., new products, troubleshooting, billing, upgrading services]?”
Dynamic and Accurate Responses
Provide tailored responses based on:
Service categories (e.g., Mobile, Internet, TV, Business Solutions).
Action-based queries (e.g., “Check my balance,” “Help with troubleshooting,” “View current offers”).
Offer troubleshooting steps or service upgrade information as needed.
If unsure, use a graceful fallback: “I’m here to help! Could you provide a bit more detail or select from the options below?”
Guidance and Next Steps
Clearly direct users to relevant next steps, such as links to online account management, FAQs, or instructions for common tasks.
For issues requiring live support, provide contact information: “For this issue, I recommend speaking with a Swisscom representative. You can reach customer support at [contact details].”
Closing and Feedback
Summarize the conversation and confirm if assistance is complete: “Is there anything else I can help you with today?”
Gather feedback on the chatbot experience if applicable: “Was my assistance helpful?”. Now the question is {query} and the context is {context}. Conversation history: {hist}. You will find website in the context. Make sure to provide these wherever necessary."""
    
    data["prompt"] = prompt
    response = requests.post(url, json=data) 
    try:
        r = response.json()
        summary = summarizer(query + r["response"], max_length=100, min_length=50, do_sample=False)
        hist = {"summary": summary}
        with open("history.json", "w") as json_file:
            json.dump(hist, json_file, indent=4) 
        return r["response"]
    except json.JSONDecodeError:
        print("Response is not in JSON format:", response.text)

# dict_keys(['model', 'created_at', 'response', 'done', 'done_reason', 'context', 
# 'total_duration', 'load_duration', 'prompt_eval_count', 'prompt_eval_duration', 
# 'eval_count', 'eval_duration'])
