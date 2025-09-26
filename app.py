import os
import tempfile
import json
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url


# Load env + configure Gemini
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing in .env")
genai.configure(api_key=API_KEY)

# Configure Cloudinary
CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUD_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUD_SECRET = os.getenv("CLOUDINARY_API_SECRET")
if not (CLOUD_NAME and CLOUD_KEY and CLOUD_SECRET):
    raise RuntimeError(
        "Cloudinary credentials missing in .env (CLOUDINARY_CLOUD_NAME/KEY/SECRET)."
    )


cloudinary.config(
    cloud_name=CLOUD_NAME,
    api_key=CLOUD_KEY,
    api_secret=CLOUD_SECRET,
)

# Choose a model (flash is fast & free-tier friendly)
MODEL_NAME = "gemini-2.5-flash"

app = FastAPI(title="Resume Analyzer (Gemini + Files)")

# Allow your frontend to call this API (adjust origin if you want)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://resume-analyser-frontend-orqc.onrender.com",
        "http://localhost:5173",
    ],  
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB - adjust if needed

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

    # Read full bytes (small resume files are ok) and enforce size limit
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    # Save upload to a temporary file (so genai.upload_file(path=...) keeps working)
    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        temp_path = tmp.name

    try:
        # ===== Upload to Cloudinary (so you get preview URL and hosted PDF) =====
        try:
            # Upload local file to Cloudinary. resource_type="auto" will accept PDFs.
            cloud_result = cloudinary.uploader.upload(
                temp_path,
                resource_type="auto",
                folder="resumes",
            )
            pdf_url = cloud_result.get("secure_url")
            public_id = cloud_result.get("public_id")

            # Build preview image (page=1 -> first page PNG)
            preview_url, _ = cloudinary_url(
                public_id,
                format="png",
                page=1,
                width=900,
                crop="scale",
            )
        except Exception as ce:
            # If Cloudinary fails, continue but include a helpful field
            pdf_url = None
            preview_url = None
            # Log or raise depending on whether Cloudinary is required
            print("Cloudinary upload failed:", str(ce))

        # ===== Upload file to Gemini (your existing flow) =====
        uploaded = genai.upload_file(path=temp_path, display_name=file.filename)

        # Build your prompt (kept your original prompt structure)
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

        # Return feedback + cloudinary urls (if available)
        return {"feedback": feedback, "pdf_url": pdf_url, "preview_url": preview_url}

    finally:
        # Clean up temp file
        try:
            os.remove(temp_path)
        except Exception:
            pass
