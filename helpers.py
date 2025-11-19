import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# Optional external libs
try:
    import docx  # python-docx
except ImportError:
    docx = None

try:
    from openai import OpenAI  # new OpenAI client
except ImportError:
    OpenAI = None  # app can still run; chat will show a warning


# ----------------- Paths & constants -----------------

BASE_DIR = Path(__file__).resolve().parent

# Your jobs JSON as in the current app
JSON_PATH = Path("/local_scrapers/local_data/job_scrapes/ai_processed_jobs.json")

# Data directory for memory + prompts
# IMPORTANT: This now uses /data by default so you can mount it as a Docker volume.
# You can override with JOB_BROWSER_DATA_DIR if you like.
DATA_DIR = Path(os.environ.get("JOB_BROWSER_DATA_DIR", "/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Memory file in DATA_DIR/job_chat_memory.json
MEMORY_PATH = DATA_DIR / "job_chat_memory.json"

# Prompt manager storage
PROMPTS_PATH = DATA_DIR / "prompts.json"

# CV directory – adapt if needed
CV_DIR = Path("/local_scrapers/local_data/chris_judkins")

# CV Creator paths
CV_SKELETON_PATH = DATA_DIR / "cv_skeleton.tex"
CV_LATEX_MEMORY_PATH = DATA_DIR / "cv_latex_memory.json"
CV_EDITABLE_LATEX_PATH = DATA_DIR / "cv_editable_latex.json"

MEMORY_WINDOW = 10
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
# Use a more capable model for LaTeX CV generation
CV_LATEX_MODEL = os.environ.get("CV_LATEX_MODEL", "gpt-4o")

# ----------------- HTML template loading -----------------

LAYOUT_PATH = BASE_DIR / "layout.html"


def _extract_template(layout_text: str, begin_marker: str, end_marker: str) -> str:
    start = layout_text.find(begin_marker)
    end = layout_text.find(end_marker)
    if start == -1 or end == -1 or end <= start:
        return ""
    start += len(begin_marker)
    return layout_text[start:end].strip()


try:
    layout_text = LAYOUT_PATH.read_text(encoding="utf-8")
    HEADER_TEMPLATE = _extract_template(
        layout_text,
        "<!-- HEADER_TEMPLATE_BEGIN -->",
        "<!-- HEADER_TEMPLATE_END -->",
    )
    METRIC_TEMPLATE = _extract_template(
        layout_text,
        "<!-- METRIC_TEMPLATE_BEGIN -->",
        "<!-- METRIC_TEMPLATE_END -->",
    )
except FileNotFoundError:
    # Minimal fallbacks if layout.html is missing
    HEADER_TEMPLATE = (
        "<div><strong>{company}</strong> — {location} — {salary} — "
        "<a href='{job_url}' target='_blank'>Open original job posting</a></div>"
    )
    METRIC_TEMPLATE = "<div><strong>{title}</strong>: {value}</div>"


def render_header_html(company: str, location: str, salary: str, job_url: str) -> str:
    return HEADER_TEMPLATE.format(
        company=company or "(no company)",
        location=location or "",
        salary=salary or "",
        job_url=job_url or "#",
    )


def render_metric_html(title: str, value: Any) -> str:
    return METRIC_TEMPLATE.format(
        title=title,
        value=value if value not in (None, "") else "/",
    )


# ----------------- Small helpers -----------------


def clean_multiline_text(value: Any) -> str:
    """
    Convert '/n' markers from the JSON into real line breaks '\n'
    and return a clean string.
    """
    if value is None:
        return ""
    text = str(value)
    return text.replace("/n", "\n")


def lines_to_bullets(text: str) -> str:
    """Turn a multi-line string into a markdown bullet list."""
    lines = [l.rstrip() for l in text.split("\n")]
    bullet_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("-", "*", "•")):
            bullet_lines.append(stripped)
        else:
            bullet_lines.append(f"- {stripped}")
    return "\n".join(bullet_lines)


def skills_list(value: Any) -> List[str]:
    """Ensure we always get a list of strings for skills columns."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
        return [p for p in parts if p]
    return [str(value)]


def parse_numeric_column(df: pd.DataFrame, col_name: str, new_col_name: str) -> None:
    """
    Parse a numeric-like column that may contain commas or % signs.
    Adds a new column with numeric values (float) or NaN.
    """
    if col_name not in df.columns:
        df[new_col_name] = pd.NA
        return

    s = (
        df[col_name]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    df[new_col_name] = pd.to_numeric(s, errors="coerce")


def normalize_sections(sections) -> List[Dict[str, str]]:
    """Return sections as a list of {title, text} dicts, with '/n' respected as line breaks."""
    if isinstance(sections, dict):
        return [
            {"title": k, "text": clean_multiline_text(v)}
            for k, v in sections.items()
        ]

    if isinstance(sections, list):
        normalized = []
        for s in sections:
            if isinstance(s, dict):
                title = s.get("title") or s.get("section") or ""
                raw_text = s.get("text") or s.get("content") or ""
                text = clean_multiline_text(raw_text)
                if not title and len(s) == 1:
                    only_key = next(iter(s))
                    title = only_key
                    text = clean_multiline_text(s[only_key])
                normalized.append({"title": title, "text": text})
            elif isinstance(s, str):
                normalized.append({"title": "", "text": clean_multiline_text(s)})
            else:
                normalized.append({"title": "", "text": clean_multiline_text(s)})
        return normalized

    if sections:
        return [{"title": "", "text": clean_multiline_text(sections)}]
    return []


# ----------------- Jobs load/save -----------------


def load_jobs(path: Path = JSON_PATH) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    data = json.loads(text)
    if isinstance(data, dict):
        jobs = list(data.values())
    elif isinstance(data, list):
        jobs = data
    else:
        jobs = []

    for job in jobs:
        if not isinstance(job, dict):
            continue

        # Normalize user_score / human_score
        if "human_score" in job:
            user_score_val = job.get("user_score")
            human_score_val = job.get("human_score")
            if (user_score_val is None or user_score_val == "") and human_score_val not in (
                None,
                "",
            ):
                job["user_score"] = human_score_val
            job.pop("human_score", None)

        job.setdefault("overwritten", False)
        job.setdefault("user_score", None)

        # New fields
        job.setdefault("application_sent", False)
        job.setdefault("user_note", "")

    return jobs


def save_jobs(path: Path, jobs: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")


# ----------------- Prompt manager helpers -----------------


def load_prompts(path: Path = PROMPTS_PATH) -> Dict[str, str]:
    """
    Load saved prompts from JSON.
    Structure: { "Prompt name": "Prompt text", ... }
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return {}
        data = json.loads(text)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
        return {}
    except Exception:
        return {}


def save_prompts(prompts: Dict[str, str], path: Path = PROMPTS_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")


# ----------------- CV + memory helpers -----------------


def load_cv_corpus(cv_dir: Path = CV_DIR) -> str:
    """
    Concatenate all local .docx CV files into one big text blob.
    If python-docx or the directory is missing, returns empty string.
    """
    if "cv_corpus" in st.session_state:
        return st.session_state["cv_corpus"]

    if docx is None or not cv_dir.exists():
        st.session_state["cv_corpus"] = ""
        return ""

    texts: List[str] = []
    for docx_path in sorted(cv_dir.glob("*.docx")):
        try:
            doc = docx.Document(str(docx_path))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            if paragraphs:
                texts.append(
                    f"--- CV file: {docx_path.name} ---\n" + "\n".join(paragraphs)
                )
        except Exception as e:
            texts.append(f"--- Failed to read {docx_path.name}: {e} ---")

    corpus = "\n\n".join(texts)
    st.session_state["cv_corpus"] = corpus
    return corpus


def load_memory(path: Path = MEMORY_PATH) -> Dict[str, Any]:
    """
    Load per-job chat memory from disk.
    Structure: { url: {"messages": [ {role, content}, ... ] } }
    """
    path = Path(path)

    if "chat_memory" in st.session_state:
        return st.session_state["chat_memory"]

    if not path.exists():
        memory: Dict[str, Any] = {}
        st.session_state["chat_memory"] = memory
        return memory

    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            memory = {}
        else:
            memory = json.loads(text)
    except Exception:
        memory = {}

    for url, entry in list(memory.items()):
        msgs = entry.get("messages", [])
        if not isinstance(msgs, list):
            memory[url] = {"messages": []}

    st.session_state["chat_memory"] = memory
    return memory


def save_memory(memory: Dict[str, Any], path: Path = MEMORY_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")


def get_job_memory(url: str) -> Dict[str, Any]:
    memory = load_memory()
    if url not in memory:
        memory[url] = {"messages": []}
    return memory[url]


def truncate_memory(entry: Dict[str, Any], window: int = MEMORY_WINDOW) -> None:
    msgs = entry.get("messages", [])
    max_msgs = window * 2
    if len(msgs) > max_msgs:
        entry["messages"] = msgs[-max_msgs:]


def build_job_context(job: Dict[str, Any], cover_letter_text: str) -> str:
    title = job.get("title", "")
    company = job.get("company", "")
    location = job.get("location", "")
    salary = job.get("salary", "")

    matching_skills = skills_list(job.get("matching_skills"))
    missing_skills = skills_list(job.get("missing_skills"))

    analysis_raw = job.get("job_matching_analysis") or ""
    analysis = clean_multiline_text(analysis_raw)

    sections = normalize_sections(job.get("sections"))

    lines: List[str] = []
    lines.append(f"Job title: {title}")
    lines.append(f"Company: {company}")
    lines.append(f"Location: {location}")
    if salary:
        lines.append(f"Salary: {salary}")
    lines.append("")
    if matching_skills:
        lines.append("Matching skills (user already has):")
        for s in matching_skills:
            lines.append(f"- {s}")
        lines.append("")
    if missing_skills:
        lines.append("Missing/weak skills for this job:")
        for s in missing_skills:
            lines.append(f"- {s}")
        lines.append("")
    if analysis:
        lines.append("Job matching analysis (how well user fits):")
        lines.append(analysis)
        lines.append("")
    if sections:
        lines.append("Job offer sections (shortened):")
        for sec in sections[:5]:
            t = sec["title"] or "(no heading)"
            text = sec["text"]
            if len(text) > 600:
                text = text[:600] + " ..."
            lines.append(f"=== {t} ===")
            lines.append(text)
            lines.append("")
    if cover_letter_text:
        lines.append("Current custom cover letter shown in the UI:")
        lines.append(cover_letter_text)

    return "\n".join(lines)


# ----------------- Chat model wrapper -----------------


def call_chat_model(
    job: Dict[str, Any],
    cover_letter_text: str,
    user_prompt: str,
    url: str,
    extra_docs: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Call OpenAI chat completion with:
    - job context
    - current cover letter text
    - user CV texts
    - short per-job memory (last turns)
    - optional extra user-provided documents (name + content)
    """
    if OpenAI is None:
        return (
            "⚠️ OpenAI Python client not installed. Install `openai>=1.0` "
            "inside your environment."
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return (
            "⚠️ Environment variable OPENAI_API_KEY is not set. "
            "Please configure it and restart the app."
        )

    client = OpenAI(api_key=api_key)

    cv_text = load_cv_corpus()
    job_context = build_job_context(job, cover_letter_text)

    system_content = (
        "You are an AI assistant helping the user evaluate ONE specific job offer "
        "and craft application materials. Always focus on the CURRENT job offer only.\n\n"
        "Tasks:\n"
        "- Answer detailed questions about the job based on the provided job description.\n"
        "- Suggest improvements or variants for the user's cover letter.\n"
        "- Propose bullet points for emails or notes related to this job.\n\n"
        "Inputs you receive:\n"
        "1) Structured description of the current job offer.\n"
        "2) Current custom cover letter for this job.\n"
        "3) Text extracted from the user's CV documents.\n"
        "4) A short memory of previous turns for THIS job profile only.\n"
        "5) Optionally, extra user-provided documents, each with a custom name.\n\n"
        "Constraints:\n"
        "- Be concise and practical.\n"
        "- If you modify the cover letter, keep a professional tone and do NOT invent facts "
        "that are not in the CV or extra-document context.\n"
        "- Reply in the same language as the user's prompt (German or English).\n\n"
        "IMPORTANT:\n"
        "- You DO have access to the contents of all 'Additional user-provided documents' "
        "listed in the context (for example a document named 'Smart Agent Config.docx').\n"
        "- When the user refers to such a document by name, assume it is exactly the document "
        "with that name in the additional-documents section, and use its content.\n"
        "- Never claim that you do not have access to these additional documents. Instead, base "
        "your reasoning and answers on their content."
    )

    memory_entry = get_job_memory(url)
    truncate_memory(memory_entry, MEMORY_WINDOW)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_content},
        {
            "role": "system",
            "content": (
                "Context about the CURRENT job and cover letter:\n\n"
                + job_context
                + "\n\nContext from the user's CV documents:\n\n"
                + (cv_text or "(No CV documents loaded.)")
            ),
        },
    ]

    # Add extra user documents as separate system context (if any)
    if extra_docs:
        docs_lines: List[str] = []
        for d in extra_docs:
            name = d.get("name", "Unnamed document")
            content = d.get("content", "") or ""
            if not content:
                continue
            # Truncate very long docs for safety
            if len(content) > 4000:
                content_short = content[:4000] + " ..."
            else:
                content_short = content
            docs_lines.append(f"=== {name} ===\n{content_short}")
        if docs_lines:
            messages.append(
                {
                    "role": "system",
                    "content": "Additional user-provided documents:\n\n"
                    + "\n\n".join(docs_lines),
                }
            )

    # Memory of previous turns
    for msg in memory_entry.get("messages", []):
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current user prompt
    messages.append({"role": "user", "content": user_prompt})

    completion = client.chat.completions.create(
        model=DEFAULT_OPENAI_MODEL,
        messages=messages,
        temperature=0.4,
    )

    answer = completion.choices[0].message.content.strip()

    # Update memory with final user+assistant turn
    memory_entry["messages"].append({"role": "user", "content": user_prompt})
    memory_entry["messages"].append({"role": "assistant", "content": answer})
    truncate_memory(memory_entry, MEMORY_WINDOW)

    memory = load_memory()
    memory[url] = memory_entry
    save_memory(memory)

    return answer


# ----------------- CV Creator helpers -----------------


def load_cv_skeleton(path: Path = CV_SKELETON_PATH) -> str:
    """
    Load the LaTeX CV skeleton template from disk.
    Returns empty string if not found.
    """
    path = Path(path)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def save_cv_skeleton(skeleton: str, path: Path = CV_SKELETON_PATH) -> None:
    """Save the LaTeX CV skeleton template to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(skeleton, encoding="utf-8")


def load_cls_files(data_dir: Path = DATA_DIR) -> Dict[str, str]:
    """
    Load all .cls files from the data directory.
    Returns a dict mapping filename to content.
    """
    cls_files = {}
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return cls_files

    for cls_path in data_dir.glob("*.cls"):
        try:
            cls_files[cls_path.name] = cls_path.read_text(encoding="utf-8")
        except Exception as e:
            cls_files[cls_path.name] = f"% ERROR reading {cls_path.name}: {e}"

    return cls_files


def load_latex_memory(path: Path = CV_LATEX_MEMORY_PATH) -> Dict[str, Any]:
    """
    Load per-job LaTeX CV chat memory from disk.
    Structure: { url: {"messages": [ {role, content}, ... ] } }
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return {}
        memory = json.loads(text)
        # Ensure proper structure
        for url, entry in list(memory.items()):
            msgs = entry.get("messages", [])
            if not isinstance(msgs, list):
                memory[url] = {"messages": []}
        return memory
    except Exception:
        return {}


def save_latex_memory(memory: Dict[str, Any], path: Path = CV_LATEX_MEMORY_PATH) -> None:
    """Save LaTeX CV chat memory to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")


def get_latex_job_memory(url: str) -> Dict[str, Any]:
    """Get or create LaTeX memory entry for a specific job."""
    memory = load_latex_memory()
    if url not in memory:
        memory[url] = {"messages": []}
    return memory[url]


def load_editable_latex(path: Path = CV_EDITABLE_LATEX_PATH) -> Dict[str, str]:
    """
    Load per-job editable LaTeX content from disk.
    Structure: { url: "latex_content" }
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return {}
        return json.loads(text)
    except Exception:
        return {}


def save_editable_latex(latex_dict: Dict[str, str], path: Path = CV_EDITABLE_LATEX_PATH) -> None:
    """Save per-job editable LaTeX content to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(latex_dict, ensure_ascii=False, indent=2), encoding="utf-8")


def call_latex_cv_model(
    job: Dict[str, Any],
    cv_skeleton: str,
    user_prompt_addition: str,
    default_prompt: str,
    language: str,
    url: str,
    editable_latex: str,
) -> str:
    """
    Call OpenAI chat completion for LaTeX CV generation with:
    - CV skeleton template
    - Job context
    - User CV texts
    - Language preference (German/English)
    - Default prompt (rules for CV generation)
    - User prompt addition (custom instructions)
    - Current editable LaTeX content (so model can reference it)
    - Short per-job memory (last 10 turns)

    Returns LaTeX code only (enforced via system prompt).
    """
    if OpenAI is None:
        return (
            "% ERROR: OpenAI Python client not installed. Install `openai>=1.0`."
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return (
            "% ERROR: Environment variable OPENAI_API_KEY is not set."
        )

    client = OpenAI(api_key=api_key)

    cv_text = load_cv_corpus()
    job_context = build_job_context(job, "")  # No cover letter needed for CV
    cls_files = load_cls_files()  # Load any .cls files from /data

    # Build the full prompt from default + user addition
    full_user_prompt = default_prompt
    if user_prompt_addition.strip():
        full_user_prompt += "\n\nAdditional instructions:\n" + user_prompt_addition

    # System prompt enforcing LaTeX-only output with STRICT structure preservation
    system_content = (
        f"You are an expert LaTeX CV generator. Your task is to create a professional, "
        f"one-page CV in LaTeX based on the provided CV skeleton template.\n\n"
        f"⚠️ CRITICAL STRUCTURAL REQUIREMENTS - DO NOT VIOLATE THESE:\n"
        f"1. PRESERVE the \\documentclass line EXACTLY as provided in the skeleton\n"
        f"2. PRESERVE all \\usepackage commands EXACTLY as provided\n"
        f"3. PRESERVE the overall document structure (sections, subsections, environments)\n"
        f"4. ONLY replace placeholder text in square brackets like [YOUR NAME], [Job Title], etc.\n"
        f"5. DO NOT add new packages unless absolutely necessary for the content\n"
        f"6. DO NOT change the documentclass or class options\n"
        f"7. If the skeleton uses a custom class file (e.g., altacv.cls), keep it unchanged\n\n"
        f"✅ WHAT YOU CAN MODIFY:\n"
        f"- Replace all placeholder content in square brackets with actual information\n"
        f"- Add or remove bullet points within existing itemize environments\n"
        f"- Adjust spacing slightly (\\vspace) to fit content on one page\n"
        f"- Duplicate existing section patterns if more entries are needed (e.g., more jobs)\n\n"
        f"📋 CONTENT REQUIREMENTS:\n"
        f"1. Output ONLY valid LaTeX code - no explanations, no markdown code fences\n"
        f"2. The CV MUST be exactly 1 page when compiled\n"
        f"3. Write the entire CV in {language} language\n"
        f"4. Select and emphasize information from the user's CV that BEST matches the job\n"
        f"5. Follow professional CV best practices\n"
        f"6. Tailor content to highlight the most relevant qualifications for this job\n\n"
        f"7. Never make any information about the user up - only use what is in the CV text and other documents\n\n"
        f"8. Never mix up work experience with personal projects or education, and ensure that you implement all work experience entries from my original CV\n\n"
        f"9. You may add or delete bullet points within existing sections to best fit the one-page requirement and the importance of this section for the current job offer\n\n"
        f"🔧 TECHNICAL NOTES:\n"
        f"- If .cls files are provided, they define the document class and must be referenced correctly\n"
        f"- Maintain exact command names and environments from the skeleton\n"
        f"- Keep the same formatting style throughout\n\n"
        f"- Ensure that the document starts with '\begin{{filecontents*}}{{page1sidebar.tex}}' if present in the skeleton\n\n"
        f"💡 EDITING EXISTING CVs:\n"
        f"If you see 'Current editable LaTeX code', the user may ask you to modify it. "
        f"When editing, maintain the EXACT structure and only change the content as requested.\n\n"
        f"⚠️ OUTPUT FORMAT: Return ONLY the complete LaTeX code, nothing else. No explanations."
    )

    memory_entry = get_latex_job_memory(url)
    truncate_memory(memory_entry, MEMORY_WINDOW)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_content},
    ]

    # Add .cls files if they exist
    if cls_files:
        cls_content_parts = []
        for filename, content in cls_files.items():
            cls_content_parts.append(f"=== {filename} ===\n{content}")
        messages.append({
            "role": "system",
            "content": (
                "LaTeX Class Files (.cls) available in /data directory:\n\n"
                + "\n\n".join(cls_content_parts)
                + "\n\n"
                + "IMPORTANT: If the CV skeleton uses one of these class files (e.g., \\documentclass{altacv}), "
                + "you MUST keep the \\documentclass line exactly as specified in the skeleton. "
                + "These class files are available and will be used during compilation."
            ),
        })

    # Add CV skeleton, job context, and user CV
    messages.append({
        "role": "system",
        "content": (
            "CV Skeleton Template (PRESERVE THIS STRUCTURE EXACTLY):\n\n"
            + (cv_skeleton or "(No CV skeleton provided.)")
            + "\n\n---\n\n"
            + "Job Profile to tailor CV for:\n\n"
            + job_context
            + "\n\n---\n\n"
            + "User's CV and personal information:\n\n"
            + (cv_text or "(No CV documents loaded.)")
        ),
    })

    # Add current editable LaTeX as context
    if editable_latex.strip():
        messages.append(
            {
                "role": "system",
                "content": (
                    "Current editable LaTeX code (user can reference this):\n\n"
                    + editable_latex
                ),
            }
        )

    # Memory of previous turns
    for msg in memory_entry.get("messages", []):
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current user prompt
    messages.append({"role": "user", "content": full_user_prompt})

    completion = client.chat.completions.create(
        model=CV_LATEX_MODEL,
        messages=messages,
        temperature=0.3,  # Lower temperature for more consistent LaTeX output
    )

    answer = completion.choices[0].message.content.strip()

    # Clean up any markdown code fences if present (```latex ... ```)
    if answer.startswith("```"):
        lines = answer.split("\n")
        # Remove first line if it's a code fence
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        # Remove last line if it's a code fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        answer = "\n".join(lines)

    # Update memory with final user+assistant turn
    memory_entry["messages"].append({"role": "user", "content": full_user_prompt})
    memory_entry["messages"].append({"role": "assistant", "content": answer})
    truncate_memory(memory_entry, MEMORY_WINDOW)

    memory = load_latex_memory()
    memory[url] = memory_entry
    save_latex_memory(memory)

    return answer
