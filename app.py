from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import os
from dotenv import load_dotenv
import pdfplumber
from werkzeug.utils import secure_filename
from rag_store import add_to_store, query_store
from openai import OpenAI, AzureOpenAI
import re

# Azure AI Foundry SDK
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

#Load environment variables
load_dotenv()

#flask setup
app = Flask(__name__)
app.secret_key = "supersecret" # replace for production


# Uploads
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "txt"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Azure DeepSeek setup
DEEPSEEK_ENDPOINT = os.getenv("AZURE_DEEPSEEK_ENDPOINT")
DEEPSEEK_KEY = os.getenv("AZURE_DEEPSEEK_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

# Fallback if Azure is not configured
USE_DEEPSEEK = all([DEEPSEEK_ENDPOINT, DEEPSEEK_KEY, DEPLOYMENT_NAME])
if USE_DEEPSEEK:
    from openai import OpenAI
    deepseek_client = OpenAI(base_url=DEEPSEEK_ENDPOINT, api_key=DEEPSEEK_KEY)
else:
    print("⚠️ Azure DeepSeek not configured. Using simulated responses.")
    
    
if not DEEPSEEK_ENDPOINT or not DEEPSEEK_KEY or not DEPLOYMENT_NAME:
    raise ValueError("Missing Azure DeepSeek configuration in .env")


deepseek_client = OpenAI(
    base_url=f"{DEEPSEEK_ENDPOINT}",
    api_key=DEEPSEEK_KEY
)

# Helpers
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_file(filepath):
    """Extract raw text from PDF or TXT."""
    text = ""
    if filepath.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    elif filepath.endswith(".pdf"):
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
    return text


# Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        flash("No file uploaded")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected")
        return redirect(url_for("index"))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        raw_text = extract_text_from_file(filepath)
        if not raw_text.strip():
            flash("Failed to extract text from file")
            return redirect(url_for("index"))

        add_to_store(raw_text, book_name=filename)
        flash(f"Book '{filename}' uploaded successfully!")
        return redirect(url_for("index"))
    else:
        flash("Invalid file type. Please upload .pdf or .txt")
        return redirect(url_for("index"))

@app.route("/chat", methods=["GET"])
def chat_get():
    if "chat_history" not in session:
        session["chat_history"] = []
        session["character"] = ""

    character_selected = False
    new_character = request.args.get("character", "").strip()

    if new_character and new_character != session.get("character", ""):
        session["character"] = new_character
        session["chat_history"] = []  # clear previous conversation
        character_selected = True

    return render_template(
        "chat.html",
        chat_history=session.get("chat_history", []),
        character=session.get("character") or "Unknown Character"
        )

@app.route("/chat", methods=["POST"])
def chat_post():
    user_input = request.form.get("user_input", "").strip()
    new_character = request.form.get("character", "").strip()

    if new_character and new_character != session.get("character", ""):
        session["character"] = new_character
        session["chat_history"] = []  # clear previous conversation

    character = session.get("character") or "Unknown Character"

    if user_input:
        context_chunks = query_store(user_input, top_k=5)
        context_text = "\n".join(context_chunks)

        conversation_text = "You: Hello\n"
        for role, msg in session["chat_history"]:
            speaker = character if role == "bot" else "You"
            conversation_text += f"{speaker}: {msg}\n"

        system_prompt = f"""
            You are {character}, a fictional character from the Harry Potter series.  

            Your ONLY job is to speak exactly as {character} would in direct dialogue.  

            Strict Rules:
            - Respond ONLY with spoken dialogue.  
            - Do NOT include thoughts, reasoning, narration, or description.  
            - Do NOT write stage directions, body language,  internal monologue, explanations, or commentary.  
            - Do NOT explain what you are doing or why.  
            - Do NOT act as an AI or assistant.  
            - DO NOT mention the rules or that you are following instructions.
            - Do NOT invent new facts outside the Harry Potter books.  
            - If you truly would not know, simply say so in character.

            If the user asks something outside {character}'s knowledge, respond as {character} would :
                - Show confusion, irritation, or dismissiveness.
                - You may say things like "I don’t know what you’re talking about," or "That is nonsense," or stay silent.
                - Never break character or explain why you don’t know.
            
            Fallback:
                - If you have no relevant knowledge from the books, still reply in short, in-character spoken words (not narration).
                - Never output nothing; always give a natural, character-like response.

            The user is a regular person, not a Harry Potter character.  

            Context to stay consistent:
            - Previous conversation:
            {conversation_text}

            - Current user message:
            {user_input}

            - Book context:
            {context_text}

            Important:
            - Your output must be *only* what {character} says out loud.  
            - No narration. No explanations. No tags. No formatting. No commentary.  
            - If this is the first message, do NOT introduce yourself or explain your role — just respond in character with dialogue.  

            Final Rule: If your output contains anything other than spoken dialogue, you have failed the task.  
            """

        if USE_DEEPSEEK:
            try:
                response = deepseek_client.chat.completions.create(
                    model=DEPLOYMENT_NAME,
                    messages=[{"role": "user", "content": system_prompt}],
                    temperature=0.7,
                )
                bot_reply = response.choices[0].message.content.strip()
                if "<think>" in bot_reply and "</think>" in bot_reply:
                    bot_reply = re.sub(r"<think>.*?</think>", "", bot_reply, flags=re.DOTALL).strip()

                # Optional: keep only the first quoted dialogue if the model still adds fluff
                if '"' in bot_reply:
                    first_quote = bot_reply.find('"')
                    last_quote = bot_reply.rfind('"')
                    if last_quote > first_quote:
                        bot_reply = bot_reply[first_quote+1:last_quote]  # text between the quotes
            except Exception as e:
                print("⚠️ DeepSeek API error:", e)
                bot_reply = f"(Error using DeepSeek. Context: {context_text[:200]}...)"
        else:
            bot_reply = f"(Simulated reply based on context: {context_text[:200]}...)"

        chat_history = session.get("chat_history", [])
        chat_history.append(("user", user_input))
        chat_history.append(("bot", bot_reply))
        session["chat_history"] = chat_history[-5:]

    return render_template(
        "chat.html",
        chat_history=session.get("chat_history", []),
        character=session.get("character") or "Unknown Character"
    )


if __name__ == "__main__":
    app.run(debug=True)
