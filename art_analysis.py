import anthropic
import base64
import json
import re
import sys
import os
from dotenv import load_dotenv

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

if __name__ == "__main__":
    image_path = sys.argv[1]
    result = analyze_sketch(image_path)
    print(json.dumps(result, indent=2))