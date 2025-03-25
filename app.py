import openai
import os
import uuid
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist
static_dir = "static"
os.makedirs(static_dir, exist_ok=True)

# Create templates directory if it doesn't exist
templates_dir = "templates"
os.makedirs(templates_dir, exist_ok=True)

# Serve static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize templates
templates = Jinja2Templates(directory=templates_dir)

# Get OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

allowed_keywords = [
    "EY", "Cypress", "Detectron2", "ChromaDB", "resume-JD analyzer",
    "Python", "SQL", "Machine Learning", "internship", "AI", "ML"
]

# Pydantic models
class QuestionRequest(BaseModel):
    question: str

class ResponseModel(BaseModel):
    text: str
    audio_url: str | None = None

# Helper functions
def get_chatgpt_response(prompt: str) -> str:
    """
    Get a response from ChatGPT based on Nikhil Bhandari's professional profile.
    
    Args:
        prompt: User's input/question
        
    Returns:
        str: Generated response based on Nikhil's profile
        
    Raises:
        HTTPException: If there's an error with the OpenAI API
    """
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # System message with structured profile information
        system_content = """
        You are Nikhil Bhandari, an AI/ML Engineer with expertise in:
        - Programming: Python, SQL
        - Technologies: Computer Vision, NLP, Agentic AI
        - Visualization: Power BI, QlikSense
        - Cloud: Amazon EC2
        - Frameworks: FastAPI
        
        Professional Background:
        - Former EY intern passionate about Data Science
        - Worked on diverse projects including:
          * Image segmentation with Detectron2 and UNet
          * Text-to-PPT converter
          * Multi-agent AI system using CrewAI for persona extraction
          * BERTopic-based Reddit topic modeling and visualization
        
        Common Misconceptions:
        - Some assume you're only technical, but you actively contribute to strategy and business impact
        
        Current Goals:
        1) Deepen Data Science expertise through industry-standard projects
        2) Master MLOps, containerization, cloud AI, and real-time inference
        3) Develop diverse, production-ready AI applications
        
        Response Guidelines:
        - Only respond based on this context
        - If unsure: "I don't have information on that"
        - Keep responses conversational (2-3 sentences)
        - Maintain professional yet approachable tone
        """

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            temperature=0,  # For consistent, factual responses
            messages=[
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate response: {str(e)}"
        )

def validate_response(response: str) -> bool:
    return any(keyword.lower() in response.lower() for keyword in allowed_keywords)

def text_to_speech(text: str) -> str:
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        
        filename = f"{uuid.uuid4().hex}.mp3"
        filepath = os.path.join(static_dir, filename)
        
        with open(filepath, "wb") as f:
            f.write(response.content)
            
        return filename
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS Error: {str(e)}")

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask", response_model=ResponseModel)
def handle_question(request: QuestionRequest):
    try:
        # Get ChatGPT response
        response_text = get_chatgpt_response(request.question)
        
        # Validate response
        if not validate_response(response_text):
            response_text = "I don't have information on that."
        
        # Generate audio
        audio_filename = text_to_speech(response_text)
        return ResponseModel(
            text=response_text,
            audio_url=f"/static/{audio_filename}"
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    #uvicorn app:app --reload