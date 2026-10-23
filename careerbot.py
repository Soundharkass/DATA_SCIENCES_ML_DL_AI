import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
SYSTEM_PROMPT = """You are CareerPilot, a practical AI career and technology learning agent.
Help users understand technology, explore tech jobs, learn skills, and prepare for interviews.
When a user asks for a career learning roadmap, call build_learning_plan and then explain the result.
Ask a short follow-up only if the target role is missing. You do not have live web search in this free-tier
version: do not claim that you checked current job listings, salaries, or recent news. Be clear when advice
may vary by location or change over time. Never claim to have applied for a job or contacted anyone.
Do not ask for passwords, API keys, government ID numbers, or sensitive personal details.
Keep answers friendly, clear, and useful to a beginner. Make no guarantees about employment.
"""

TOOLS = [
    {
        "type": "function",
        "name": "build_learning_plan",
        "description": "Create a practical week-by-week learning plan for a technology job goal.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_role": {"type": "string", "description": "The technology job the user wants."},
                "current_skills": {"type": "string", "description": "Skills or experience the user already has."},
                "hours_per_week": {"type": "integer", "description": "Study hours available each week, from 1 to 40."},
                "weeks": {"type": "integer", "description": "Plan length, from 2 to 16 weeks."},
            },
            "required": ["target_role", "current_skills", "hours_per_week", "weeks"],
        },
    }
]


def build_learning_plan(target_role: str, current_skills: str, hours_per_week: int, weeks: int) -> dict[str, Any]:
    """Create a bounded plan outline for Gemini to tailor and explain."""
    hours = max(1, min(int(hours_per_week), 40))
    duration = max(2, min(int(weeks), 16))
    role = target_role.strip()[:120] or "entry-level technology role"
    skills = current_skills.strip()[:500] or "Beginner; no skills listed yet"
    week_hours = max(1, hours // 4)

    milestones = []
    for week in range(1, duration + 1):
        if week == 1:
            focus = "Check common requirements for the role and assess your starting skills"
        elif week == duration:
            focus = "Finish a portfolio project, review gaps, and practice explaining your work"
        elif week >= duration - 1:
            focus = "Build and polish a practical portfolio project related to the role"
        else:
            focus = "Learn one core skill and complete a small hands-on exercise"
        milestones.append({"week": week, "focus": focus, "suggested_hours": week_hours})

    return {
        "target_role": role,
        "starting_skills": skills,
        "hours_per_week": hours,
        "duration_weeks": duration,
        "milestones": milestones,
    }


def safe_error_details(exc: Exception) -> str:
    detail = re.sub(r"AIza[A-Za-z0-9_-]{20,}", "[redacted Gemini API key]", str(exc))
    detail = " ".join(detail.split())[:400]
    return f"{type(exc).__name__}: {detail or 'No additional details were provided.'}"


def run_agent(messages: list[dict[str, str]]) -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise RuntimeError("GEMINI_API_KEY is missing. Add your Google AI Studio key to .env, then restart the app.")

    client = genai.Client(api_key=api_key)
    transcript = "\n".join(
        f"{'User' if item['role'] == 'user' else 'CareerPilot'}: {item['content']}"
        for item in messages[-12:]
    )
    interaction = client.interactions.create(
        model=MODEL,
        system_instruction=SYSTEM_PROMPT,
        tools=TOOLS,
        input=transcript,
    )

    # Execute local plan tools and return the result to Gemini for its final response.
    for _ in range(5):
        calls = [step for step in interaction.steps if getattr(step, "type", "") == "function_call"]
        if not calls:
            break

        function_results = []
        for call in calls:
            if call.name != "build_learning_plan":
                result = {"error": "Unknown tool"}
            else:
                try:
                    result = build_learning_plan(**dict(call.arguments))
                except (TypeError, ValueError) as exc:
                    result = {"error": f"The tool could not use those arguments: {exc}"}
            function_results.append({
                "type": "function_result",
                "name": call.name,
                "call_id": call.id,
                "result": [{"type": "text", "text": json.dumps(result)}],
            })

        interaction = client.interactions.create(
            model=MODEL,
            system_instruction=SYSTEM_PROMPT,
            tools=TOOLS,
            previous_interaction_id=interaction.id,
            input=function_results,
        )

    answer = (interaction.output_text or "I couldn't create a response. Please try again.").strip()
    return answer


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        return jsonify({"error": "Send at least one chat message."}), 400

    clean_messages = []
    for item in messages[-20:]:
        if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
            return jsonify({"error": "Chat messages must have a user or assistant role."}), 400
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            return jsonify({"error": "Chat messages cannot be empty."}), 400
        clean_messages.append({"role": item["role"], "content": content[:5000]})

    if clean_messages[-1]["role"] != "user":
        return jsonify({"error": "The last message must be from the user."}), 400

    try:
        return jsonify({"reply": run_agent(clean_messages)})
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Gemini CareerPilot request failed")
        return jsonify({"error": f"Gemini request failed. {safe_error_details(exc)} Check the PowerShell window for details."}), 502


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
