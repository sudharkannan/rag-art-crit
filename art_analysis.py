import anthropic
import base64
import json
import re
import sys
import os
from dotenv import load_dotenv
from retrieval import retrieve_for_analysis

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) 

def analyze_sketch(image_path, subject=None):
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    ext = image_path.rsplit(".", 1)[-1].lower()
    media_type_map = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png", "gif": "image/gif", "webp": "image/webp"
    }
    media_type = media_type_map.get(ext, "image/jpeg")

    subject_context = f"The artist describes this as: {subject}\n" if subject else ""

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": f"""{subject_context}Analyze this sketch for common beginner drawing issues, as well as possible improvements that could be made.

Do not flag as errors:
- Deliberate design choices visible in the overall piece
- Deliberate stylistic choices

If unsure whether something is deliberate or a mistake, mention it in the response.

Optionally include 1-2 general_tips based on apparent skill level. 
Can be empty array if nothing relevant comes to mind.

While being encouraging is a plus, remember that the focus is on specific, honest, and valid critiques.

Return ONLY valid JSON, no other text:
{{
  "issues": [
    {{
      "category": "proportions",
      "observation": "specific thing you see in this sketch",
      "severity": "high"
    }}
  ],
  "suggestions": [
    {{
      "category": "appeal",
      "observation": "specific opportunity you see in this sketch",
      "impact": "high",
    }}
  ],
  "general_tips": []
}}

Limit to top 2-3 issues and suggestions.
Error Categories: proportions, foreshortening, line_confidence, shading, perspective, composition.
Suggestion categories: storytelling, appeal, lighting, detail_balance, style_consistency, general

Severity: high, medium, low."""
                }
            ]
        }]
    )

    raw = response.content[0].text
    cleaned = re.sub(r"```json|```", "", raw).strip()
    return json.loads(cleaned)

def synthesize_feedback(analysis, context_docs, image_data, media_type, subject=None):
    subject_context = f"The artist describes this as: {subject}\n" if subject else ""
    
    context_text = "\n\n---\n\n".join(context_docs) if context_docs else "No additional context available."

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": f"""{subject_context}You are an experienced art teacher giving feedback to a beginner artist.

Here is the analysis of their sketch:
{json.dumps(analysis, indent=2)}

Here is relevant instructional content to draw from:
---
{context_text}
---

Write encouraging but honest feedback that:
- Addresses each issue and suggestion from the analysis
- Explains WHY each problem occurs, not just that it exists
- Gives one concrete exercise or fix per issue
- Suggests specific search terms or resources where relevant
- Focuses on clarity and remains concise and specific.

Write in plain conversational language as a teacher would. Do not use JSON. 
Do not number every point rigidly — write it as natural flowing feedback."""
                }
            ]
        }]
    )

    return response.content[0].text

def full_critique(image_path, subject=None):
    print("Analyzing sketch...")
    analysis = analyze_sketch(image_path, subject)
    
    print("Retrieving relevant knowledge...")
    context_docs = retrieve_for_analysis(analysis)
    
    print("Synthesizing feedback...")
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    ext = image_path.rsplit(".", 1)[-1].lower()
    media_type_map = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png", "gif": "image/gif", "webp": "image/webp"
    }
    media_type = media_type_map.get(ext, "image/jpeg")
    
    feedback = synthesize_feedback(analysis, context_docs, image_data, media_type, subject)
    return analysis, feedback

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze.py <image_path> [subject description]")
        sys.exit(1)

    image_path = sys.argv[1]
    subject = sys.argv[2] if len(sys.argv) > 2 else None

    analysis, feedback = full_critique(image_path, subject)
    
    print("\n=== ANALYSIS ===")
    print(json.dumps(analysis, indent=2))
    print("\n=== FEEDBACK ===")
    print(feedback)