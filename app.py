import os
import tempfile
import json
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai

# Load env + configure Gemini
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing in .env")
genai.configure(api_key=API_KEY)

# Choose a model (flash is fast & free-tier friendly)
MODEL_NAME = "gemini-1.5-flash"

app = FastAPI(title="Resume Analyzer (Gemini + Files)")

# Allow your frontend to call this API (adjust origin if you want)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_resume(
    file: UploadFile = File(...),
    company: str = Form(""),
    title: str = Form(""),
    description: str = Form("")
):
    # Validate content type (PDF/DOC/DOCX)
    allowed = {
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    if file.content_type not in allowed:
        raise HTTPException(status_code=415, detail="Only PDF/DOC/DOCX are supported")

    # Save upload to a temporary file
    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name

    try:
        # Upload file to Gemini
        uploaded = genai.upload_file(path=temp_path, display_name=file.filename)

        # Build your prompt (corrected version)
        prompt = f"""
interface_Feedback {{
  overallScore: number; //max 100
  ATS: {{
    score: number; //rate based on ATS suitability
    tips: {{
      type: "good" | "improve";
      tip: string; //give 3-4 tips
    }}[];
  }};
  toneAndStyle: {{
    score: number; //max 100
    tips: {{
      type: "good" | "improve";
      tip: string; //make it a short "title" for the actual explanation
      explanation: string; //explain in detail here
    }}[]; //give 3-4 tips
  }};
  content: {{
    score: number; //max 100
    tips: {{
      type: "good" | "improve";
      tip: string; //make it a short "title" for the actual explanation
      explanation: string; //explain in detail here
    }}[]; //give 3-4 tips
  }};
  structure: {{
    score: number; //max 100
    tips: {{
      type: "good" | "improve";
      tip: string; //make it a short "title" for the actual explanation
      explanation: string; //explain in detail here
    }}[]; //give 3-4 tips
  }};
  skills: {{
    score: number; //max 100
    tips: {{
      type: "good" | "improve";
      tip: string; //make it a short "title" for the actual explanation
      explanation: string; //explain in detail here
    }}[]; //give 3-4 tips
  }};
}}

You are an expert in ATS (Applicant Tracking System) and resume analysis.
Please analyze and rate this resume and suggest how to improve it.
The rating can be low if the resume is bad.
Be thorough and detailed. Don't be afraid to point out any mistakes or areas for improvement.
If there is a lot to improve, don't hesitate to give low scores. This is to help the user to improve their resume.
If available, use the job description for the job user is applying to to give more detailed feedback.
If provided, take the job description into consideration.
The company name is: {company}
The job title is: {title}
The job description is: {description}
Provide the feedback using the following format: AIResponseFormat
Return the analysis as a JSON object, without any other text and without the backticks. 
Do not include any other text or comments."""

        model = genai.GenerativeModel(MODEL_NAME)
        resp = model.generate_content(
            [uploaded, prompt],
            generation_config={"response_mime_type": "application/json"}
        )

        text = resp.text or ""
        try:
            feedback = json.loads(text)
        except Exception:
            # If the model returns non-JSON, send raw text so you can debug
            feedback = {"raw": text}

        return {"feedback": feedback }

    finally:
        # Clean up temp file
        try:
            os.remove(temp_path)
        except Exception:
            pass