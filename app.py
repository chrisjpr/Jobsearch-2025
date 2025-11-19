from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from helpers import (
    JSON_PATH,
    OpenAI,
    clean_multiline_text,
    lines_to_bullets,
    load_jobs,
    save_jobs,
    parse_numeric_column,
    skills_list,
    normalize_sections,
    render_header_html,
    render_metric_html,
    call_chat_model,
    get_job_memory,
    load_memory,
    save_memory,
    load_prompts,
    save_prompts,
    load_cv_skeleton,
    save_cv_skeleton,
    load_latex_memory,
    save_latex_memory,
    get_latex_job_memory,
    load_editable_latex,
    save_editable_latex,
    call_latex_cv_model,
)

# For reading additional uploaded .docx files
try:
    import docx  # python-docx
except ImportError:
    docx = None

load_dotenv()

# ----------------- UI config -----------------

st.set_page_config(
    page_title="Job Browser",
    layout="wide",
)

# Global small CSS helpers (for chat answer box etc.)
st.markdown(
    """
<style>
.assistant-answer-box {
    border-radius: 8px;
    padding: 0.75rem 0.9rem;
    margin-top: 0.5rem;
    margin-bottom: 0.75rem;
    background-color: rgba(148, 163, 184, 0.18); /* slate-ish, works in light/dark */
    border: 1px solid rgba(148, 163, 184, 0.7);
}
</style>
""",
    unsafe_allow_html=True,
)

# Limit page optionally
limit_page = False
if limit_page:
    st.markdown(
        """
    <style>
    .block-container {
        max-width: 80%;
        margin: auto;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

if "flash_msg" in st.session_state:
    st.success(st.session_state.pop("flash_msg"))

jobs = load_jobs(JSON_PATH)

if not jobs:
    st.warning(f"No jobs found. Make sure JSON exists at:\n`{JSON_PATH}`")
    st.stop()

df = pd.json_normalize(jobs)

# Column normalization / initialization
if "human_score" in df.columns:
    # Extra safety merge at DF level
    if "user_score" not in df.columns:
        df["user_score"] = df["human_score"]
    else:
        mask = (
            df["user_score"].isna()
            | (df["user_score"] == "")
            | (df["user_score"].astype(str) == "None")
        )
        df.loc[mask, "user_score"] = df.loc[mask, "human_score"]
    df = df.drop(columns=["human_score"])

if "overwritten" not in df.columns:
    df["overwritten"] = False
if "user_score" not in df.columns:
    df["user_score"] = pd.NA
if "application_sent" not in df.columns:
    df["application_sent"] = False
if "user_note" not in df.columns:
    df["user_note"] = ""

parse_numeric_column(df, "skills_match", "skills_match__num")
parse_numeric_column(df, "interests_match", "interests_match__num")
parse_numeric_column(df, "user_score", "user_score__num")

# Title
st.title("Job Browser")

# ----------------- Sidebar: selection card placeholder + filters -----------------

# Placeholder that will later be filled with the current-selection card,
# but is positioned here so it appears at the very top of the sidebar.
selection_card_placeholder = st.sidebar.empty()

st.sidebar.header("Filters")

search_query = st.sidebar.text_input("Search in title/company/location", "")

# Source filter disabled → always show all sources
search_key_col = "search_key"
search_keys = (
    sorted(df[search_key_col].dropna().unique())
    if search_key_col in df.columns
    else []
)

with st.sidebar.expander("Search keys", expanded=False):
    selected_search_keys = st.multiselect(
        "Search key",
        search_keys,
        default=search_keys,
    )

# Min skills_match
min_skills = None
if "skills_match__num" in df.columns:
    s = df["skills_match__num"]
    if s.notna().any():
        min_val = float(s.min(skipna=True))
        max_val = float(s.max(skipna=True))
    else:
        min_val, max_val = 0.0, 1.0
    if min_val == max_val:
        min_val_adj = min_val - 0.1
        max_val_adj = max_val + 0.1
    else:
        min_val_adj, max_val_adj = min_val, max_val
    min_skills = st.sidebar.slider(
        "Min skills_match",
        min_value=min_val_adj,
        max_value=max_val_adj,
        value=min_val_adj,
        step=(max_val_adj - min_val_adj) / 100 if max_val_adj > min_val_adj else 0.01,
    )

# Min interests_match
min_interests = None
if "interests_match__num" in df.columns:
    s2 = df["interests_match__num"]
    if s2.notna().any():
        min_val2 = float(s2.min(skipna=True))
        max_val2 = float(s2.max(skipna=True))
    else:
        min_val2, max_val2 = 0.0, 1.0
    if min_val2 == max_val2:
        min_val2_adj = min_val2 - 0.1
        max_val2_adj = max_val2 + 0.1
    else:
        min_val2_adj, max_val2_adj = min_val2, max_val2
    min_interests = st.sidebar.slider(
        "Min interests_match",
        min_value=min_val2_adj,
        max_value=max_val2_adj,
        value=min_val2_adj,
        step=(max_val2_adj - min_val2_adj) / 100 if max_val2_adj > min_val2_adj else 0.01,
    )

# Min user_score
min_user_score = None
if "user_score__num" in df.columns:
    s3 = df["user_score__num"]
    valid_scores = s3.dropna()
    if not valid_scores.empty:
        umin = float(valid_scores.min())
        umax = float(valid_scores.max())
        if umin == umax:
            umin_adj = umin - 1.0
            umax_adj = umax + 1.0
        else:
            umin_adj, umax_adj = umin, umax
        min_user_score = st.sidebar.slider(
            "Min user_score",
            min_value=umin_adj,
            max_value=umax_adj,
            value=float(0),
            step=(umax_adj - umin_adj) / 100 if umax_adj > umin_adj else 0.1,
        )
    else:
        min_user_score = None

# ----------------- Apply filters -----------------

filtered = df.copy()

if search_keys and selected_search_keys:
    filtered = filtered[filtered[search_key_col].isin(selected_search_keys)]

if search_query:
    q = search_query.lower()
    mask = (
        filtered.get("title", "").astype(str).str.lower().str.contains(q)
        | filtered.get("company", "").astype(str).str.lower().str.contains(q)
        | filtered.get("location", "").astype(str).str.lower().str.contains(q)
    )
    filtered = filtered[mask]

if min_skills is not None:
    filtered = filtered[filtered["skills_match__num"].fillna(0) >= min_skills]

if min_interests is not None:
    filtered = filtered[filtered["interests_match__num"].fillna(0) >= min_interests]

if min_user_score is not None:
    filtered = filtered[filtered["user_score__num"].fillna(0) >= min_user_score]

st.markdown(f"**Showing {len(filtered)} of {len(df)} jobs**")

# Tabs: add Custom cover letter as its own tab
overview_tab, detail_tab, cover_tab, chat_tab, cv_creator_tab, note_tab = st.tabs(
    ["Overview", "Detailed Job Profile", "Cover Letter", "Chat Assistant", "CV Creator", "User note"]
)

#region OVERVIEW TAB
# ----------------- OVERVIEW TAB -----------------

with overview_tab:
    # Column order: title first, rest as before
    base_cols = [
        "user_score",
        "overwritten",
        "application_sent",
        "title",
        "company",
        "salary",
        "skills_match",
        "interests_match",
        "location",
        "fetchedAt",
        "source",
        "search_key",
        "adage",
        "url",
    ]
    display_cols: List[str] = []
    if "title" in filtered.columns:
        display_cols.append("title")
    for col in base_cols:
        if col != "title" and col in filtered.columns:
            display_cols.append(col)

    st.subheader("Job list")

    if display_cols:
        table_df = filtered[display_cols].copy()
    else:
        table_df = filtered.copy()

    # Clean salary for display (strip trailing "/Jahr (geschätzt für Vollzeit)")
    if "salary" in table_df.columns:
        table_df["salary"] = (
            table_df["salary"]
            .astype(str)
            .str.replace("/Jahr (geschätzt für Vollzeit)", "", regex=False)
        )

    # Clean / format fetchedAt column
    if "fetchedAt" in table_df.columns:
        table_df["fetchedAt"] = (
            pd.to_datetime(table_df["fetchedAt"], errors="coerce")
            .dt.strftime("%Y-%m-%d %H:%M:%S")
        )

    # Hidden original index for mapping edits back to jobs list
    table_df["__orig_idx"] = filtered.index

    # --- Selection: keep current selection in session and mirror it into a checkbox column ---
    selected_idx: Optional[int] = st.session_state.get("selected_job_idx")

    # On first load (or if None) → default to first filtered row
    if selected_idx is None and len(filtered) > 0:
        selected_idx = int(filtered.index[0])
        st.session_state["selected_job_idx"] = selected_idx

    # Add 'selected' checkbox column, mark the currently selected job as True
    table_df["selected"] = False
    if selected_idx is not None and selected_idx in filtered.index:
        table_df.loc[table_df["__orig_idx"] == selected_idx, "selected"] = True

    # Editable table (user_score + checkboxes + selection)
    edited_df = st.data_editor(
        table_df,
        use_container_width=True,
        hide_index=True,
        key="jobs_table",
        column_order=["selected"] + display_cols + ["__orig_idx"],
        column_config={
            "selected": st.column_config.CheckboxColumn(
                "selected",
                help="Tick the row that should be the current selection.",
                default=False,
            ),
            "user_score": st.column_config.NumberColumn(
                "user_score",
                help="Your personal rating for this job (e.g., 0–10).",
                step=0.1,
            ),
            "overwritten": st.column_config.CheckboxColumn(
                "overwritten",
                help="Mark jobs where you manually adjusted the cover letter or scoring.",
                default=False,
            ),
            "application_sent": st.column_config.CheckboxColumn(
                "application_sent",
                help="Tick if you already submitted an application for this job.",
                default=False,
            ),
            "__orig_idx": st.column_config.NumberColumn(
                "__orig_idx",
                help="Internal index (hidden).",
                disabled=True,
                width="small",
            ),
        },
    )

    # Persist inline edits back to jobs JSON + update selection
    updated_any = False
    selected_job_indices: List[int] = []

    if not edited_df.empty and "__orig_idx" in edited_df.columns:
        for _, row in edited_df.iterrows():
            try:
                job_idx = int(row["__orig_idx"])
            except Exception:
                continue
            if not (0 <= job_idx < len(jobs)):
                continue

            job = jobs[job_idx]

            # user_score parsing
            new_score_raw = row.get("user_score", None)
            if pd.isna(new_score_raw):
                new_score_val: Optional[float] = None
            else:
                try:
                    new_score_val = float(new_score_raw)
                except Exception:
                    new_score_val = None

            new_overwritten = bool(row.get("overwritten", False))
            new_app_sent = bool(row.get("application_sent", False))

            # Track which rows are selected
            if bool(row.get("selected", False)):
                selected_job_indices.append(job_idx)

            if (
                job.get("user_score") != new_score_val
                or bool(job.get("overwritten", False)) != new_overwritten
                or bool(job.get("application_sent", False)) != new_app_sent
            ):
                job["user_score"] = new_score_val
                job["overwritten"] = new_overwritten
                job["application_sent"] = new_app_sent
                updated_any = True

    # Update JSON if anything changed
    if updated_any:
        try:
            save_jobs(JSON_PATH, jobs)
        except Exception as e:
            st.error(f"Failed to save inline changes: {e}")

    # Update current selection based on 'selected' checkboxes (only one, newest)
    prev_selected = st.session_state.get("selected_job_idx")
    if selected_job_indices:
        new_selected = prev_selected
        # Prefer a newly ticked index that differs from previous selection
        for idx in selected_job_indices:
            if idx != prev_selected:
                new_selected = idx
                break
        st.session_state["selected_job_idx"] = new_selected
    else:
        # If nothing is selected, treat as "no selection"
        st.session_state["selected_job_idx"] = None

# ----------------- DETAILED JOB PROFILE TAB -----------------

with detail_tab:
    st.subheader("Detailed Job Profile")

    selected_idx: Optional[int] = st.session_state.get("selected_job_idx")

    if selected_idx is None or not (0 <= selected_idx < len(jobs)):
        st.info("Please select a job in the Overview tab first.")
    else:
        job = jobs[selected_idx]

        st.markdown(f"## {job.get('title', '(no title)')}")

        raw_salary = str(job.get("salary", "") or "")
        # Strip the long suffix
        salary_display = raw_salary.replace(
            "/Jahr (geschätzt für Vollzeit)", ""
        ).strip()

        company_display = job.get("company", "")
        location_display = job.get("location", "")
        job_url = job.get("url") or ""

        # Header card from HTML template (incl. open original job link)
        header_html = render_header_html(
            company=company_display,
            location=location_display,
            salary=salary_display,
            job_url=job_url,
        )
        st.markdown(header_html, unsafe_allow_html=True)

        # Extra salary highlight
        st.markdown(
            f"""
<div style="
    margin-top:0.75rem;
    margin-bottom:0.75rem;
    display:inline-block;
    padding:0.4rem 0.9rem;
    border-radius:999px;
    border:2px solid #0b7285;
    background-color:#e3fafc;
    font-weight:600;
    font-size:0.95rem;
">
💰 Salary: {salary_display or "n/a"}
</div>
""",
            unsafe_allow_html=True,
        )

        # Metric cards (skills & interests)
        score_col1, score_col2 = st.columns(2)
        skills_val = job.get("skills_match", "")
        interests_val = job.get("interests_match", "")

        with score_col1:
            skills_html = render_metric_html("Skills match", skills_val)
            st.markdown(skills_html, unsafe_allow_html=True)

        with score_col2:
            interests_html = render_metric_html("Interests match", interests_val)
            st.markdown(interests_html, unsafe_allow_html=True)

        # ----------------- Editable form (status & job info) -----------------
        with st.form("job_detail_form"):
            # Status & evaluation section (three fields side by side)
            st.markdown("### Status & evaluation")
            status_col1, status_col2, status_col3 = st.columns(3)

            with status_col1:
                overwritten_default = bool(job.get("overwritten", False))
                overwritten = st.checkbox(
                    "Manually overwritten",
                    value=overwritten_default,
                    help="Mark jobs where you manually adjusted the cover letter or scoring.",
                )

            with status_col2:
                user_score_default = job.get("user_score")
                if user_score_default is None or (
                    isinstance(user_score_default, float)
                    and pd.isna(user_score_default)
                ):
                    user_score_default_str = ""
                else:
                    try:
                        user_score_default_str = f"{float(user_score_default):.2f}"
                    except Exception:
                        user_score_default_str = str(user_score_default)

                user_score_str = st.text_input(
                    "User score",
                    value=user_score_default_str,
                    help="Your personal rating for this job (e.g., 0–10). Leave empty for no score.",
                )

            with status_col3:
                application_sent_default = bool(job.get("application_sent", False))
                application_sent = st.checkbox(
                    "Application sent",
                    value=application_sent_default,
                    help="Tick if you already submitted an application for this job.",
                )

            st.markdown("---")

            # Tabs for Matching Analysis, Matching Skills, Sections
            matching = skills_list(job.get("matching_skills"))
            missing = skills_list(job.get("missing_skills"))
            sections = normalize_sections(job.get("sections"))
            analysis_raw = job.get("job_matching_analysis") or ""
            analysis = clean_multiline_text(analysis_raw)

            tab_analysis, tab_skills, tab_sections = st.tabs(
                ["Matching analysis", "Matching skills", "Sections"]
            )

            with tab_analysis:
                if analysis:
                    bullet_analysis = lines_to_bullets(analysis)
                    st.markdown(bullet_analysis)
                else:
                    st.write("No job matching analysis available.")

            with tab_skills:
                c1, c2 = st.columns(2)
                if matching:
                    c1.write("**Matching skills:**")
                    for s in matching:
                        c1.markdown(f"- {s}")
                if missing:
                    c2.write("**Missing skills:**")
                    for s in missing:
                        c2.markdown(f"- {s}")
                if not matching and not missing:
                    st.write("No skills information available.")

            with tab_sections:
                if sections:
                    sec_tabs = st.tabs(
                        [
                            sec["title"] or f"Section {i+1}"
                            for i, sec in enumerate(sections)
                        ]
                    )
                    for sec_tab, sec in zip(sec_tabs, sections):
                        with sec_tab:
                            bullet_text = lines_to_bullets(sec["text"])
                            st.markdown(bullet_text)
                else:
                    st.write("No sections available.")

            # Save button
            submitted = st.form_submit_button("Save changes")

        if submitted:
            # Parse user_score from text input
            if not user_score_str.strip():
                user_score_val: Optional[float] = None
            else:
                try:
                    user_score_val = float(user_score_str.replace(",", "."))
                except Exception:
                    user_score_val = None
            job["overwritten"] = bool(overwritten)
            job["user_score"] = user_score_val
            job["application_sent"] = bool(application_sent)

            try:
                save_jobs(JSON_PATH, jobs)
                st.session_state["flash_msg"] = f"Saved changes for job: {job.get('title', '(no title)')}"
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save changes: {e}")

#region COVER LETTER TAB
# ----------------- CUSTOM COVER LETTER TAB -----------------

with cover_tab:
    st.subheader("Cover Letter")

    selected_idx: Optional[int] = st.session_state.get("selected_job_idx")

    if selected_idx is None or not (0 <= selected_idx < len(jobs)):
        st.info("Please select a job in the Overview tab first.")
    else:
        job = jobs[selected_idx]
        job_url = job.get("url") or selected_idx

        # Keep cover-letter text in session per job so chat sees latest unsaved edits
        cover_state_key = f"cover_letter_text::{job_url}"
        cover_default = clean_multiline_text(job.get("custom_cover_letter"))
        if cover_state_key not in st.session_state:
            st.session_state[cover_state_key] = cover_default

        current_cover_text = st.session_state[cover_state_key]
        num_lines = max(3, current_cover_text.count("\n") + 1)
        text_height = min(800, 40 + 20 * num_lines)

        with st.form("cover_letter_form"):
            st.markdown(
                "Edit the cover letter used for this job. "
                "The chat assistant always sees the CURRENT text shown here."
            )
            st.text_area(
                "Cover letter",
                height=text_height,
                key=cover_state_key,
            )
            cover_submitted = st.form_submit_button("Save cover letter")

        if cover_submitted:
            job["custom_cover_letter"] = st.session_state[cover_state_key]
            try:
                save_jobs(JSON_PATH, jobs)
                st.success("Cover letter saved.")
            except Exception as e:
                st.error(f"Failed to save cover letter: {e}")

#region CHAT ASSISTANT TAB
# ----------------- CHAT ASSISTANT TAB -----------------

with chat_tab:
    st.subheader("LLM Assistance")

    selected_idx: Optional[int] = st.session_state.get("selected_job_idx")

    if selected_idx is None or not (0 <= selected_idx < len(jobs)):
        st.info("Please select a job in the Overview tab first.")
    else:
        job = jobs[selected_idx]
        job_url = job.get("url") or f"job-{selected_idx}"

        # Ensure cover_letter text is in state (shared with cover_tab)
        cover_state_key = f"cover_letter_text::{job_url}"
        cover_default = clean_multiline_text(job.get("custom_cover_letter"))
        if cover_state_key not in st.session_state:
            st.session_state[cover_state_key] = cover_default
        current_cover_letter = st.session_state.get(cover_state_key, cover_default)

        if OpenAI is None:
            st.warning(
                "The OpenAI Python client is not installed. "
                "Install it in your environment with `pip install openai>=1.0` to enable the chat assistant."
            )
        else:
            # Load prompts for prompt manager
            prompts_dict = load_prompts()
            prompt_names = sorted(prompts_dict.keys())

            job_memory_key = job_url
            job_memory = get_job_memory(job_memory_key)
            messages = job_memory.get("messages", [])

            # Extra docs state (per job)
            extra_docs_key = f"extra_docs::{job_memory_key}"
            if extra_docs_key not in st.session_state:
                st.session_state[extra_docs_key] = {}
            docs_dict: Dict[str, Dict[str, str]] = st.session_state[extra_docs_key]
            extra_docs_list = list(docs_dict.values())

            # --- Chat input & preset application logic (must happen BEFORE widgets) ---
            chat_input_key = f"chat_input::{job_memory_key}"
            chat_button_key = f"chat_button::{job_memory_key}"
            prompt_select_key = f"prompt_select::{job_memory_key}"
            preset_apply_key = f"prompt_preset_to_apply::{job_memory_key}"

            # Ensure defaults in session_state
            if chat_input_key not in st.session_state:
                st.session_state[chat_input_key] = ""
            if prompt_select_key not in st.session_state:
                st.session_state[prompt_select_key] = "(No preset)"

            # If a preset was requested to be applied, update the input BEFORE creating widgets
            if preset_apply_key in st.session_state:
                pending_name = st.session_state.pop(preset_apply_key)
                preset_text = prompts_dict.get(pending_name, "")
                st.session_state[chat_input_key] = preset_text
                # Reset dropdown to "(No preset)" so it doesn't re-apply
                st.session_state[prompt_select_key] = "(No preset)"

            # Chat input area
            user_prompt = st.text_area(
                "Your question or instruction",
                key=chat_input_key,
                placeholder="Ask about this job, request a new cover letter version, ask for pros/cons, etc.",
                height=100,
            )

            cols_chat = st.columns([1, 1, 2])
            with cols_chat[0]:
                send_clicked = st.button("Ask assistant", key=chat_button_key)
            with cols_chat[1]:
                clear_clicked = st.button(
                    "Clear memory for this job",
                    key=f"clear_mem::{job_memory_key}",
                    help="Deletes all saved conversation history for this job profile.",
                )
            with cols_chat[2]:
                prompt_options = ["(No preset)"] + prompt_names

                def _on_preset_change():
                    selected = st.session_state.get(prompt_select_key, "(No preset)")
                    if selected and selected != "(No preset)":
                        # Mark this preset to be applied on the NEXT rerun,
                        # before any widgets are created.
                        st.session_state[preset_apply_key] = selected

                selected_prompt_name = st.selectbox(
                    "Prompt preset",
                    options=prompt_options,
                    key=prompt_select_key,
                    on_change=_on_preset_change,
                )

            if clear_clicked:
                memory = load_memory()
                if job_memory_key in memory:
                    del memory[job_memory_key]
                save_memory(memory)
                st.success("Cleared chat memory for this job.")
                job_memory = get_job_memory(job_memory_key)
                messages = job_memory.get("messages", [])

            answer: Optional[str] = None
            if send_clicked and st.session_state.get(chat_input_key, "").strip():
                user_prompt_clean = st.session_state[chat_input_key].strip()
                with st.spinner("Querying chat model..."):
                    answer = call_chat_model(
                        job=job,
                        cover_letter_text=current_cover_letter,
                        user_prompt=user_prompt_clean,
                        url=job_memory_key,
                        extra_docs=extra_docs_list,
                    )
                st.session_state[f"last_answer::{job_memory_key}"] = answer

            if answer is None:
                answer = st.session_state.get(f"last_answer::{job_memory_key}")

            if answer:
                st.markdown("**Assistant answer:**")
                st.markdown(
                    f"""
<div class="assistant-answer-box">
{answer}
</div>
""",
                    unsafe_allow_html=True,
                )

            # Memory section (still in its own expander)
            with st.expander(
                "Conversation memory for this job (compact)", expanded=False
            ):
                if not messages:
                    st.write("No memory stored yet for this job.")
                else:
                    for i, msg in enumerate(messages, start=1):
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        if not content:
                            continue
                        safe_content = str(content).replace("\n", "<br>")
                        if role == "user":
                            st.markdown(
                                f'<div><span style="color:#2563eb; font-weight:600;">YOU ({i})</span>: '
                                f'{safe_content}</div>',
                                unsafe_allow_html=True,
                            )
                        elif role == "assistant":
                            st.markdown(
                                f'<div><span style="color:#16a34a; font-weight:600;">ASSISTANT ({i})</span>: '
                                f'{safe_content}</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f'<div><em>{role}</em>: {safe_content}</div>',
                                unsafe_allow_html=True,
                            )

            # Additional file upload section UNDER the memory
            st.markdown("### Additional documents for this job (optional)")
            uploaded_files = st.file_uploader(
                "Upload .docx documents to include in the assistant's context",
                type=["docx"],
                accept_multiple_files=True,
                key=f"upload::{job_memory_key}",
                help=(
                    "You can add extra documents (e.g. project descriptions, additional CVs). "
                    "Each will be included in the assistant's context for this job."
                ),
            )

            # Add newly uploaded files to session_state (by filename)
            if uploaded_files:
                for uf in uploaded_files:
                    doc_id = uf.name
                    if doc_id not in docs_dict:
                        if docx is not None:
                            try:
                                doc_obj = docx.Document(uf)
                                paragraphs = [
                                    p.text
                                    for p in doc_obj.paragraphs
                                    if p.text.strip()
                                ]
                                content = "\n".join(paragraphs)
                            except Exception as e:
                                content = f"(Failed to read {uf.name}: {e})"
                        else:
                            content = (
                                "(python-docx is not installed; cannot read .docx content.)"
                            )

                        docs_dict[doc_id] = {
                            "name": uf.name,
                            "content": content,
                        }
                st.session_state[extra_docs_key] = docs_dict

            # Show and allow renaming & deleting of stored extra documents
            if docs_dict:
                st.markdown("**Documents currently included in context:**")
                to_delete: List[str] = []
                for doc_id, doc_info in docs_dict.items():
                    cols_doc = st.columns([4, 1])
                    with cols_doc[0]:
                        new_name = st.text_input(
                            "Document name",
                            value=doc_info["name"],
                            key=f"extra_name::{job_memory_key}::{doc_id}",
                            help="Give this document a descriptive name (e.g. 'Teaching portfolio', 'Project X description').",
                        )
                        doc_info["name"] = new_name
                    with cols_doc[1]:
                        if st.button(
                            "Delete",
                            key=f"del_doc::{job_memory_key}::{doc_id}",
                            help="Remove this document from the assistant's context.",
                        ):
                            to_delete.append(doc_id)

                    with st.expander(f"{doc_info['name']}", expanded=False):
                        preview = doc_info["content"][:1200]
                        st.text(preview if preview else "(empty document)")

                # Apply deletions
                if to_delete:
                    for doc_id in to_delete:
                        docs_dict.pop(doc_id, None)
                    st.session_state[extra_docs_key] = docs_dict
                    st.success(
                        "Updated documents for this job. They will no longer be used in future answers."
                    )

            # ---------- Prompt manager at bottom of Chat tab ----------
            st.markdown("---")
            with st.expander("Prompt manager (custom reusable prompts)", expanded=False):
                st.markdown(
                    "Create, edit and delete custom prompts you can quickly insert into the chat input."
                )

                # Reload latest prompts (in case they changed earlier in this run)
                prompts_for_mgr = load_prompts()
                existing_names = sorted(prompts_for_mgr.keys())

                # Select existing prompt to edit/delete
                cols_mgr_top = st.columns([2, 1])
                with cols_mgr_top[0]:
                    selected_mgr_prompt = st.selectbox(
                        "Select prompt to edit/delete",
                        options=["(None selected)"] + existing_names,
                        key="prompt_mgr_select",
                    )
                with cols_mgr_top[1]:
                    if (
                        selected_mgr_prompt
                        and selected_mgr_prompt != "(None selected)"
                        and st.button(
                            "Delete selected prompt", key="prompt_mgr_delete"
                        )
                    ):
                        prompts_for_mgr.pop(selected_mgr_prompt, None)
                        save_prompts(prompts_for_mgr)
                        st.success(f"Deleted prompt: {selected_mgr_prompt}")
                        st.rerun()

                # Edit existing prompt text
                if (
                    selected_mgr_prompt
                    and selected_mgr_prompt != "(None selected)"
                ):
                    edit_key = f"prompt_mgr_edit_text::{selected_mgr_prompt}"
                    default_text = prompts_for_mgr.get(selected_mgr_prompt, "")
                    edited_text = st.text_area(
                        f"Edit text for '{selected_mgr_prompt}'",
                        value=default_text,
                        key=edit_key,
                        height=150,
                    )
                    if st.button("Save changes", key="prompt_mgr_save_edit"):
                        prompts_for_mgr[selected_mgr_prompt] = edited_text
                        save_prompts(prompts_for_mgr)
                        st.success(f"Updated prompt: {selected_mgr_prompt}")

                st.markdown("### Add new prompt")
                new_prompt_name = st.text_input(
                    "New prompt name", key="prompt_mgr_new_name"
                )
                new_prompt_text = st.text_area(
                    "New prompt text", key="prompt_mgr_new_text", height=150
                )
                if st.button("Add prompt", key="prompt_mgr_add"):
                    if not new_prompt_name.strip():
                        st.error("Please provide a name for the new prompt.")
                    elif new_prompt_name in prompts_for_mgr:
                        st.error(
                            "A prompt with that name already exists. Please choose another name."
                        )
                    else:
                        prompts_for_mgr[new_prompt_name] = new_prompt_text
                        save_prompts(prompts_for_mgr)
                        st.success(f"Added new prompt: {new_prompt_name}")
                        st.rerun()

#region CV CREATOR TAB
# ----------------- CV CREATOR TAB -----------------

with cv_creator_tab:
    st.subheader("LaTeX CV Creator")

    selected_idx: Optional[int] = st.session_state.get("selected_job_idx")

    if selected_idx is None or not (0 <= selected_idx < len(jobs)):
        st.info("Please select a job in the Overview tab first.")
    else:
        job = jobs[selected_idx]
        job_url = job.get("url") or f"job-{selected_idx}"

        if OpenAI is None:
            st.warning(
                "The OpenAI Python client is not installed. "
                "Install it in your environment with `pip install openai>=1.0` to enable the CV Creator."
            )
        else:
            # --- CV Skeleton Management ---
            cv_skeleton = load_cv_skeleton()

            with st.expander("CV Skeleton Template", expanded=False):
                st.markdown(
                    "Upload or edit the LaTeX CV skeleton template. "
                    "This template will be used as the base structure for all generated CVs."
                )
                skeleton_text = st.text_area(
                    "LaTeX CV Skeleton",
                    value=cv_skeleton,
                    height=300,
                    key="cv_skeleton_editor",
                    help="Paste your LaTeX CV template here. The model will use this structure and fill it with your information.",
                )

                col_skel1, col_skel2 = st.columns(2)
                with col_skel1:
                    if st.button("Save CV Skeleton", key="save_skeleton"):
                        save_cv_skeleton(skeleton_text)
                        st.success("CV skeleton saved successfully!")
                with col_skel2:
                    uploaded_skeleton = st.file_uploader(
                        "Or upload .tex file",
                        type=["tex"],
                        key="upload_skeleton",
                        help="Upload a .tex file to use as CV skeleton",
                    )
                    if uploaded_skeleton is not None:
                        skeleton_content = uploaded_skeleton.read().decode("utf-8")
                        save_cv_skeleton(skeleton_content)
                        st.success(f"Uploaded and saved skeleton from {uploaded_skeleton.name}")
                        st.rerun()

            st.markdown("---")

            # --- Language Selection ---
            col_lang, col_space = st.columns([1, 3])
            with col_lang:
                language = st.selectbox(
                    "CV Language",
                    options=["English", "German"],
                    key=f"cv_language::{job_url}",
                    help="The language in which the CV content should be written",
                )

            # --- Default Prompt (expandable, editable but not persistent) ---
            default_prompt_key = f"cv_default_prompt::{job_url}"
            if default_prompt_key not in st.session_state:
                st.session_state[default_prompt_key] = (
                    "Generate a professional, one-page LaTeX CV tailored to this job profile. "
                    "Use the provided CV skeleton structure and fill it with my most relevant "
                    "experience, skills, and qualifications that match this specific job. "
                    "Ensure the CV is concise, well-formatted, and highlights my strengths "
                    "for this position. Follow professional CV best practices."
                )

            with st.expander("Default Prompt (click to edit)", expanded=False):
                st.markdown(
                    "This is the base instruction given to the model. "
                    "You can edit it for this session, but changes are not saved across app restarts."
                )
                default_prompt = st.text_area(
                    "Default Prompt",
                    value=st.session_state[default_prompt_key],
                    height=150,
                    key=default_prompt_key,
                    help="Base instructions for CV generation (not persistent across restarts)",
                )

            # --- Custom Prompt Addition (always shown) ---
            custom_prompt_key = f"cv_custom_prompt::{job_url}"
            if custom_prompt_key not in st.session_state:
                st.session_state[custom_prompt_key] = ""

            st.markdown("### Custom Instructions")
            custom_prompt = st.text_area(
                "Additional instructions (optional)",
                value=st.session_state[custom_prompt_key],
                height=100,
                key=custom_prompt_key,
                placeholder="Add any specific instructions, e.g., 'emphasize my Python skills' or 'modify the education section to be more concise'",
                help="Add custom instructions to refine the CV generation",
            )

            # --- Generate Button & Clear Memory ---
            col_gen1, col_gen2 = st.columns([1, 1])
            with col_gen1:
                generate_clicked = st.button(
                    "Generate LaTeX CV",
                    key=f"generate_cv::{job_url}",
                    type="primary",
                    help="Generate a tailored LaTeX CV for this job",
                )
            with col_gen2:
                clear_latex_mem = st.button(
                    "Clear CV memory",
                    key=f"clear_latex_mem::{job_url}",
                    help="Deletes all saved conversation history for CV generation",
                )

            if clear_latex_mem:
                memory = load_latex_memory()
                if job_url in memory:
                    del memory[job_url]
                save_latex_memory(memory)
                st.success("Cleared CV generation memory for this job.")

            # --- Model Output Box ---
            latest_output_key = f"cv_latest_output::{job_url}"

            if generate_clicked:
                # Load editable LaTeX to pass as context
                editable_dict = load_editable_latex()
                editable_latex = editable_dict.get(job_url, "")

                with st.spinner("Generating LaTeX CV..."):
                    latex_output = call_latex_cv_model(
                        job=job,
                        cv_skeleton=cv_skeleton,
                        user_prompt_addition=custom_prompt,
                        default_prompt=default_prompt,
                        language=language,
                        url=job_url,
                        editable_latex=editable_latex,
                    )
                st.session_state[latest_output_key] = latex_output

            latest_output = st.session_state.get(latest_output_key, "")

            if latest_output:
                st.markdown("### Generated LaTeX Code")
                st.markdown(
                    '<div style="background-color: rgba(240, 242, 246, 0.5); '
                    'border: 1px solid rgba(0,0,0,0.1); border-radius: 8px; '
                    'padding: 1rem; margin-bottom: 1rem;">',
                    unsafe_allow_html=True,
                )
                st.code(latest_output, language="latex", line_numbers=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # Copy to editable button
                if st.button(
                    "📋 Copy to Editable LaTeX Box",
                    key=f"copy_to_editable::{job_url}",
                    help="Copy the generated LaTeX to the editable box below",
                ):
                    editable_dict = load_editable_latex()
                    editable_dict[job_url] = latest_output
                    save_editable_latex(editable_dict)
                    st.success("Copied to editable LaTeX box!")
                    st.rerun()

            st.markdown("---")

            # --- Editable LaTeX Box (persistent) ---
            st.markdown("### Editable LaTeX Code")
            st.markdown(
                "This LaTeX code is saved persistently. "
                "You can edit it manually or paste your own code. "
                "The model can see this code when you reference it in custom instructions."
            )

            editable_dict = load_editable_latex()
            editable_latex = editable_dict.get(job_url, "")

            editable_key = f"cv_editable_latex::{job_url}"
            if editable_key not in st.session_state:
                st.session_state[editable_key] = editable_latex

            with st.form(f"editable_latex_form::{job_url}"):
                edited_latex = st.text_area(
                    "Editable LaTeX",
                    value=st.session_state[editable_key],
                    height=400,
                    key=editable_key,
                    help="Edit your LaTeX code here. Changes are saved when you click 'Save Editable LaTeX'.",
                )

                col_save1, col_save2 = st.columns([1, 3])
                with col_save1:
                    save_editable = st.form_submit_button("Save Editable LaTeX")

                if save_editable:
                    editable_dict[job_url] = st.session_state[editable_key]
                    save_editable_latex(editable_dict)
                    st.success("Editable LaTeX saved!")

            st.markdown("---")

            # --- Conversation Memory ---
            with st.expander("CV Generation Memory (last 10 prompts)", expanded=False):
                latex_memory = load_latex_memory()
                job_latex_memory = latex_memory.get(job_url, {})
                messages = job_latex_memory.get("messages", [])

                if not messages:
                    st.write("No memory stored yet for this job's CV generation.")
                else:
                    for i, msg in enumerate(messages, start=1):
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        if not content:
                            continue
                        # Truncate long LaTeX output in memory display
                        display_content = content
                        if len(display_content) > 500:
                            display_content = display_content[:500] + "... [truncated]"

                        safe_content = str(display_content).replace("\n", "<br>")
                        if role == "user":
                            st.markdown(
                                f'<div><span style="color:#2563eb; font-weight:600;">PROMPT ({i})</span>: '
                                f'{safe_content}</div>',
                                unsafe_allow_html=True,
                            )
                        elif role == "assistant":
                            st.markdown(
                                f'<div><span style="color:#16a34a; font-weight:600;">GENERATED ({i})</span>: '
                                f'{safe_content}</div>',
                                unsafe_allow_html=True,
                            )

#region USER NOTE TAB
# ----------------- USER NOTE TAB -----------------

with note_tab:
    st.subheader("User note")

    selected_idx: Optional[int] = st.session_state.get("selected_job_idx")

    if selected_idx is None or not (0 <= selected_idx < len(jobs)):
        st.info("Please select a job in the Overview tab first.")
    else:
        job = jobs[selected_idx]
        job_url = job.get("url") or selected_idx

        note_state_key = f"user_note::{job_url}"
        note_default = job.get("user_note", "")

        if note_state_key not in st.session_state:
            st.session_state[note_state_key] = note_default

        current_note = st.session_state[note_state_key]
        num_lines_note = max(3, current_note.count("\n") + 1)
        note_height = min(600, 40 + 20 * num_lines_note)

        with st.form("user_note_form"):
            st.text_area(
                "Your personal note for this job",
                height=note_height,
                key=note_state_key,
            )
            note_submitted = st.form_submit_button("Save note")

        if note_submitted:
            job["user_note"] = st.session_state[note_state_key]
            try:
                save_jobs(JSON_PATH, jobs)
                st.success("User note saved.")
            except Exception as e:
                st.error(f"Failed to save note: {e}")

# ----------------- GLOBAL CURRENT SELECTION CARD (SIDEBAR, TOP LEFT) -----------------

selected_idx_global: Optional[int] = st.session_state.get("selected_job_idx")

if selected_idx_global is not None and 0 <= selected_idx_global < len(jobs):
    job = jobs[selected_idx_global]
    title = job.get("title", "(no title)")
    company = job.get("company", "")
    location = job.get("location", "")
    raw_salary = str(job.get("salary", "") or "")
    salary_clean = raw_salary.replace("/Jahr (geschätzt für Vollzeit)", "").strip()

    # Metrics
    score_val = job.get("user_score")
    try:
        score_text = (
            f"{float(score_val):.2f}"
            if score_val is not None and not pd.isna(score_val)
            else "/"
        )
    except Exception:
        score_text = "/"

    skills_val = job.get("skills_match", "/")
    interests_val = job.get("interests_match", "/")

    # Card styled to rely on theme text colors (no fixed text colors),
    # so it remains readable in both light and dark themes.
    card_html = f"""
<div style="
    border-radius: 10px;
    border: 1px solid rgba(0,0,0,0.2);
    padding: 0.75rem 0.9rem;
    margin-bottom: 1rem;
    font-size: 0.85rem;
">

  <div style="font-weight: 800; margin-bottom: 0.25rem; font-size: 1.2rem;">
    {title}
  </div>
  <div style="font-size: 0.8rem; margin-bottom: 0.5rem;">
    {company or ""}
  </div>
  <div style="font-size: 0.8rem; margin-bottom: 0.5rem;">
    {(location) if location else ""}
  </div>

  <div style="display: flex; gap: 0.5rem; margin-bottom: 0.5rem; font-size: 0.78rem;">
    <div style="flex: 1; text-align: center;">
      <div style="font-size: 0.65rem; text-transform: uppercase;">User score</div>
      <div style="font-weight: 600;">{score_text}</div>
    </div>
    <div style="flex: 1; text-align: center;">
      <div style="font-size: 0.65rem; text-transform: uppercase;">Skills</div>
      <div style="font-weight: 600;">{skills_val}</div>
    </div>
    <div style="flex: 1; text-align: center;">
      <div style="font-size: 0.65rem; text-transform: uppercase;">Interests</div>
      <div style="font-weight: 600;">{interests_val}</div>
    </div>
  </div>

  <div style="font-size: 0.8rem;">
    <span style="font-weight: 600;">Salary:</span> {salary_clean or "n/a"}
  </div>
</div>
"""
    selection_card_placeholder.markdown(card_html, unsafe_allow_html=True)
else:
    # If no selection, we simply don't show a card (placeholder remains empty)
    pass
