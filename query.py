import faiss
import numpy as np
import os
import time
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# API key loaded from environment variable
# Before running, set your key in terminal:
#   export GROQ_API_KEY="your_actual_key_here"
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise EnvironmentError(
        "GROQ_API_KEY not set.\n"
        "Run: export GROQ_API_KEY='your_key_here'\n"
        "Then re-run the script."
    )

print("Loading models...")
client = Groq(api_key=api_key)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

DB_PATH = "db/index.faiss"
CHUNKS_PATH = "db/chunks.txt"
OUTPUT_DIR = "outputs"
GROQ_MODEL = "llama-3.3-70b-versatile"  # Change this if you want a different model

# Load FAISS index and chunks once
if not os.path.exists(DB_PATH):
    raise FileNotFoundError(
        f"FAISS index not found at {DB_PATH}.\n"
        "Run ingest.py first to build the database."
    )

index = faiss.read_index(DB_PATH)

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = f.read().split("\n---\n")

print(f"Loaded {len(chunks)} chunks from database.\n")

# CORE UTILITIES
def get_embedding(text):
    """Generate normalized embedding — must match ingest.py."""
    return embed_model.encode(text, normalize_embeddings=True).astype("float32")


def safe_generate(prompt, retries=3):
    """Call Groq API with retry logic on failure."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            return response.choices[0].message.content

        except Exception as e:
            wait = 30 * (attempt + 1)  # 30s, 60s, 90s
            print(f"Error: {e}. Waiting {wait}s before retry {attempt + 1}/{retries}...")
            time.sleep(wait)

    return "API failed after all retries. Please check your key and limits."


def save_output(paper_name, task_name, content):
    """Save output to a markdown file in outputs/ directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_name = paper_name.replace(" ", "_").replace(".pdf", "")
    filename = f"{OUTPUT_DIR}/{safe_name}_{task_name}.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# {task_name}\n")
        f.write(f"**Paper:** {paper_name}\n\n")
        f.write("---\n\n")
        f.write(content)

    print(f"Output saved → {filename}")
    return filename

# SEARCH
def search(query, k=5, paper_name=None):
    """
    Search vector DB for relevant chunks.
    If paper_name is given, only return chunks from that paper.
    Fetches k*3 candidates to allow for filtering.
    """
    query_emb = np.array([get_embedding(query)])
    distances, indices = index.search(query_emb, k * 3)

    results = []
    for i in indices[0]:
        if i < 0 or i >= len(chunks):
            continue

        chunk = chunks[i]

        if paper_name:
            if paper_name.lower() in chunk.lower():
                results.append(chunk)
        else:
            results.append(chunk)

        if len(results) >= k:
            break

    return results

# MCP 1: SUMMARIZE
def summarize(paper_name):
    """Summarize a research paper in simple terms."""
    results = search(
        "problem statement approach methodology results conclusion",
        paper_name=paper_name
    )

    if not results:
        return f"No content found for '{paper_name}'. Check the paper name."

    context = "\n\n".join(results)

    prompt = f"""
You are a research assistant. Summarize the following research paper content in simple, clear terms.

Focus on:
1. What problem does this paper solve?
2. What approach or method do they use?
3. What are the key results or findings?
4. Why does this matter?

Keep it accessible — assume the reader is smart but not an expert in this field.

Paper Content:
{context[:4000]}
"""
    return safe_generate(prompt)

# MCP 2: COMPARE
def compare_papers(paper1_name, paper2_name):
    """Compare two research papers across key dimensions."""
    query = "problem methodology contributions results limitations"

    results1 = search(query, paper_name=paper1_name)
    results2 = search(query, paper_name=paper2_name)

    if not results1:
        return f"No content found for '{paper1_name}'."
    if not results2:
        return f"No content found for '{paper2_name}'."

    context1 = "\n\n".join(results1)
    context2 = "\n\n".join(results2)

    prompt = f"""
You are a research analyst. Compare these two research papers based strictly on the content provided.

Paper 1 ({paper1_name}):
{context1[:2500]}

Paper 2 ({paper2_name}):
{context2[:2500]}

Compare them across:
1. Problem being solved
2. Methodology / Approach
3. Key contributions
4. Strengths and weaknesses
5. Core differences

If information is missing for any point, say "Not enough information available."
Be analytical, not descriptive.
"""
    return safe_generate(prompt)

# MCP 3: IMPLEMENTATION STEPS
def classify_paper(context):
    """Detect paper type to tailor implementation guide."""
    context_lower = context.lower()

    if any(w in context_lower for w in ["theorem", "proof", "lemma", "undecidable", "turing",
    "imitation game", "computability", "philosophical","we may hope", "conjecture", "thought experiment","can machines think", "discrete state"]):
        return "theoretical"
    elif any(w in context_lower for w in ["dataset", "accuracy", "benchmark", "training", "evaluation"]):
        return "ml_experimental"
    elif any(w in context_lower for w in ["survey", "review", "literature"]):
        return "survey"
    elif any(w in context_lower for w in ["architecture", "pipeline", "system design"]):
        return "systems"

    return "general"


def build_implementation_prompt(paper_name, context, paper_type):
    """
    Return a prompt tailored to the paper type.
    Each type needs a different angle — forcing implementation steps
    out of a theoretical paper produces plausible-sounding nonsense.
    """

    if paper_type == "theoretical":
        return f"""
You are a research engineer bridging theory and practice.

This is a THEORETICAL paper — it does not contain direct implementation steps.
Do NOT invent steps that aren't grounded in the content.

Instead, based ONLY on the paper content below, provide:
1. What modern real-world problem does this theory map to?
2. How have researchers since implemented or operationalized these ideas? (infer from the theory, don't hallucinate)
3. What modern frameworks, tools, or algorithms are inspired by or related to this work?
4. What would a researcher need to study next to implement ideas from this paper?
5. Key open challenges this theory leaves for implementers

Paper: {paper_name}
Content:
{context[:4000]}
"""

    elif paper_type == "ml_experimental":
        return f"""
You are a senior ML engineer. Based ONLY on the research paper content below, create a practical implementation guide.

Paper: {paper_name}
Content:
{context[:4000]}

Provide:
1. Step-by-step implementation plan
2. Recommended modern tech stack (language, frameworks, hardware)
3. Model or algorithm mapping (original paper → modern equivalent)
4. Dataset requirements and preprocessing steps
5. Evaluation metrics to use
6. Key challenges and how to handle them

Be precise and actionable. Do not add information not present in the paper.
"""

    elif paper_type == "survey":
        return f"""
You are a research engineer. This is a SURVEY paper — it reviews existing work rather than proposing a new method.

Based ONLY on the content below, provide:
1. What is the core problem space this survey covers?
2. What are the main approaches or methods surveyed?
3. Which approach from the survey is most worth implementing today and why?
4. What modern tools or frameworks map to the surveyed methods?
5. What gaps in the surveyed field suggest the best implementation opportunities?

Paper: {paper_name}
Content:
{context[:4000]}
"""

    elif paper_type == "systems":
        return f"""
You are a systems architect. Based ONLY on the research paper content below, create a practical implementation guide.

Paper: {paper_name}
Content:
{context[:4000]}

Provide:
1. System architecture overview
2. Step-by-step implementation plan
3. Recommended modern tech stack
4. Key components and how they interact
5. Scalability and performance considerations
6. Key challenges and how to handle them

Be precise. Do not add information not present in the paper.
"""

    else:  # general
        return f"""
You are a senior engineer. Based ONLY on the research paper content below, create a practical implementation guide.

Paper: {paper_name}
Content:
{context[:4000]}

Provide:
1. Step-by-step implementation plan
2. Recommended modern tech stack
3. Model or algorithm mapping (original paper → modern equivalent if applicable)
4. Dataset requirements and preprocessing steps
5. Key challenges and how to handle them

Be precise and actionable. Do not add information not present in the paper.
"""


def implementation_steps(paper_name):
    """Generate a practical implementation guide from a paper."""
    results = search(
        "implementation architecture algorithm training dataset evaluation pipeline",
        paper_name=paper_name
    )

    if not results:
        return f"No content found for '{paper_name}'."

    context = "\n\n".join(results)
    paper_type = classify_paper(context)
    print(f"Detected paper type: {paper_type}")

    prompt = build_implementation_prompt(paper_name, context, paper_type)
    return safe_generate(prompt)



# MCP 4: RESEARCH GAPS
def research_gaps(paper_name):
    """Identify limitations, gaps, and future research directions."""
    results = search(
        "limitations drawbacks assumptions future work challenges weaknesses open problems",
        paper_name=paper_name,
        k=8
    )

    if not results:
        return f"No content found for '{paper_name}'."

    context = "\n\n".join(results)

    prompt = f"""
You are a critical research reviewer. Analyze the following paper content for gaps and weaknesses.

Paper Content:
{context[:4500]}

Provide a critical analysis covering:
1. Limitations of the proposed approach
2. What problems or cases are NOT addressed
3. Weak or unstated assumptions
4. Potential improvements to the method
5. Promising future research directions

IMPORTANT:
- Be critical and analytical, not descriptive
- Do NOT summarize what the paper does — focus on what it misses
- If something is not mentioned, note that as a gap itself
"""
    return safe_generate(prompt)

def main():
    print("=" * 50)
    print("  Research Paper Analysis System (Groq)")
    print(f"  Model: {GROQ_MODEL}")
    print("=" * 50)
    print("Modes:")
    print("  1 → Summarize a paper")
    print("  2 → Compare two papers")
    print("  3 → Implementation guide")
    print("  4 → Research gaps & limitations")
    print("  q → Quit")
    print("=" * 50)

    while True:
        mode = input("\nChoose mode (1/2/3/4/q): ").strip().lower()

        if mode == "q":
            print("Goodbye!")
            break

        elif mode == "1":
            paper = input("Enter paper filename (e.g. attention.pdf): ").strip()
            print("\nAnalyzing...")
            result = summarize(paper)
            print("\n" + "─" * 40)
            print(result)
            save_output(paper, "summary", result)

        elif mode == "2":
            paper1 = input("Enter first paper filename: ").strip()
            paper2 = input("Enter second paper filename: ").strip()
            print("\nComparing...")
            result = compare_papers(paper1, paper2)
            print("\n" + "─" * 40)
            print(result)
            save_output(f"{paper1}_vs_{paper2}", "comparison", result)

        elif mode == "3":
            paper = input("Enter paper filename: ").strip()
            print("\nGenerating implementation guide...")
            result = implementation_steps(paper)
            print("\n" + "─" * 40)
            print(result)
            save_output(paper, "implementation", result)

        elif mode == "4":
            paper = input("Enter paper filename: ").strip()
            print("\nIdentifying research gaps...")
            result = research_gaps(paper)
            print("\n" + "─" * 40)
            print(result)
            save_output(paper, "gaps", result)

        else:
            print("Invalid choice. Enter 1, 2, 3, 4, or q.")


if __name__ == "__main__":
    main()