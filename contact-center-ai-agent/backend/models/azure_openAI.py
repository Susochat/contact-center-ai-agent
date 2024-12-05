from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai

# Initialize FastAPI app
app = FastAPI()

# Azure OpenAI credentials
AZURE_OPENAI_API_KEY = "your_access_token"
AZURE_OPENAI_ENDPOINT = "your_endpoint"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o-mini"  # Update with your deployment name

# Configure the OpenAI library for Azure
openai.api_type = "azure"
openai.api_key = AZURE_OPENAI_API_KEY
# openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = "2024-10-21"
openai.azure_endpoint=AZURE_OPENAI_ENDPOINT

# Define intent labels
INTENT_LABELS = {
    0: "Order Management",
    1: "Technical Support",
    2: "Account & Billing",
    3: "Product Inquiry",
    4: "General Inquiry & Feedback"
}

# Pydantic model for input and output
class ConversationInput(BaseModel):
    history: list  # Expecting a list of messages with sender and text

class AnalysisResult(BaseModel):
    summary: str
    sentiment: str
    intent: str
    suggestions: str

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_conversation(conversation: ConversationInput):
    """
    Endpoint to analyze the entire chat history.
    Args:
        conversation (ConversationInput): JSON input with full chat history.
    Returns:
        AnalysisResult: JSON response with summary, sentiment, intent, and suggestions.
    """
    try:
        # Prepare the conversation history as a single string for analysis
        history_text = "\n".join(
            f"{msg['sender'].capitalize()}: {msg['text']}" for msg in conversation.history
        )

        # Build the prompt
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional AI assistant skilled in analyzing "
                    "contact center conversations. Your goal is to provide actionable insights and "
                    "specific, step-by-step suggestions to assist the agent in delivering excellent customer service."
                ),
            },
            {
                "role": "user",
                "content": f"""
                Analyze the following conversation history:

                {history_text}

                Tasks:
                1. Summarize the conversation.
                2. Assess the customer's sentiment (Positive, Neutral, Negative).
                3. Identify the customer's intent using these categories:
                   0: Order Management
                   1: Technical Support
                   2: Account & Billing
                   3: Product Inquiry
                   4: General Inquiry & Feedback
                4. Provide actionable next steps for the agent, including:
                   - Acknowledgment or Greetings whichever if applicable.
                   - Clear steps to resolve the issue. [Format - Steps to Resolve: 1. example , 2. example ..]
                   - Additional suggestions to enhance customer satisfaction. [Format - Additional Suggestions: 1. example , 2. example ..]
                    Format should be like this: 
                    Acknowledgment: Generated Acknowledgemet/Greetings  (this should be a heading and end with new line)
                    Steps to Resolve: (convert list of strings to strings, this should be a heading and should end with new line)
                    1. example . 
                    2. example .. (\n)
                    Additional Suggestions: (convert list of strings to strings , this should be a heading and should end with new line)
                    1. example . 
                    2. example .. (\n)

                Return the results in JSON format:
                {{
                    "summary": "Summary of the conversation",
                    "sentiment": "Positive | Neutral | Negative",
                    "intent": {{ "id": <number>, "description": "<category>" }},
                    "suggestions": "Detailed and professional next steps"
                }}
                """
            },
        ]

        # Call Azure OpenAI API
        response = openai.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=messages,
            max_tokens=500,
            temperature=0.7,
        )

        # Parse the assistant's response
        result_text = response.choices[0].message.content.strip('```json\n').strip('```')
        result = eval(result_text)  # Convert JSON-like string to Python dictionary

        # Map intent ID to human-readable label
        intent_id = result["intent"]["id"]
        intent_description = INTENT_LABELS.get(intent_id, "Unknown Intent")

        # Ensure suggestions is a string
        if isinstance(result['suggestions'], dict):
            result['suggestions'] = ' '.join([f"{key}: {value}" for key, value in result['suggestions'].items()])

        return AnalysisResult(
            summary=result["summary"],
            sentiment=result["sentiment"],
            intent=f"{intent_description}",
            suggestions=result["suggestions"]  # This should now be a string
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """
    Root endpoint for API health check.
    """
    return {"message": "Contact Center AI Assistant API is running!"}
