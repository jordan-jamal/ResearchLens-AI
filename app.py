import streamlit as st
import fitz
import os
import time
import json
import re
import html
import textwrap
from dotenv import load_dotenv
from google import genai

st.set_page_config(
    page_title="ResearchLens AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def load_css(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )
load_css("style.css")

load_dotenv()

# Support both local .env and Streamlit Cloud secrets
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    st.error(
        "Gemini API key not found. "
        "Add GEMINI_API_KEY to your .env file (local) "
        "or to Secrets in Streamlit Cloud settings."
    )
    st.stop()
client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-3.5-flash-lite"

def extract_text_from_pdf(pdf_file):
    document = fitz.open(
        stream=pdf_file.read(),
        filetype="pdf"
    )
    text = ""
    for page in document:
        page_text = page.get_text()
        if page_text:
            text += page_text + "\n"
    document.close()
    return text

def clean_text(text):
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+"," ", text)
    return text.strip()

def chunk_text(text, chunk_size=10000, overlap=500):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == text_length:
            break
        start = end - overlap
    return chunks

def call_gemini(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )
            if response.text:
                return response.text
            return "No response generated."
        except Exception as e:
            error_message = str(e)
            # Rate limit error
            if "429" in error_message:
                if attempt < max_retries - 1:
                    wait_time = 20 * (attempt + 1)
                    time.sleep(wait_time)
                    continue
            raise e


def answer_question(question, paper_text):
    """
    Answer a user's question using the uploaded research paper.
    For long papers, select the most relevant text chunks.
    """

    if not paper_text or not paper_text.strip():
        return "Please upload a research paper first."

    chunks = chunk_text(paper_text)

    question_words = set(
        re.findall(r"\b[a-zA-Z]{3,}\b", question.lower())
    )

   
    scored_chunks = []

    for index, chunk in enumerate(chunks):
        chunk_words = set(
            re.findall(r"\b[a-zA-Z]{3,}\b", chunk.lower())
        )

        score = len(question_words & chunk_words)

        scored_chunks.append((score, index, chunk))

    
    scored_chunks.sort(key=lambda item: item[0], reverse=True)

    relevant_chunks = [
        item for item in scored_chunks[:4] if item[0] > 0
    ]

    if not relevant_chunks:
        relevant_chunks = [
            (0, index, chunk)
            for index, chunk in enumerate(chunks[:3])
        ]

    context = "\n\n".join(
        f"Text section {index + 1}:\n{chunk}"
        for score, index, chunk in relevant_chunks
    )

    prompt = f"""
You are ResearchLens AI, an assistant that answers questions
about academic research papers.

Answer the user's question using ONLY the paper excerpts below.

Instructions:
1. Give a clear, accurate answer in simple language.
2. Do not invent facts, results, statistics, or citations.
3. If the excerpts do not contain enough information, say so.
4. Distinguish what the paper explicitly states from any inference.
5. Use bullet points when they improve clarity.
6. Do not claim to have checked the entire paper if the excerpts
   do not contain the required information.

PAPER EXCERPTS:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    try:
        response = call_gemini(prompt)
        return response

    except Exception as e:
        return (
            "Sorry, I couldn't answer that question right now. "
            f"Please try again. Error: {str(e)}"
        )


def summarize_paper(text):
    prompt = f"""
    You are ResearchLens AI, an expert research paper summarization system.
    Your task is to create a structured and faithful summary of the
    research paper below.
    STRICT FAITHFULNESS RULES:

    1. Use ONLY information present in the research paper.
    2. Do NOT invent facts.
    3. Do NOT assume missing information.
    4. Preserve numerical values exactly.
    5. Preserve important experimental results exactly.
    6. Do NOT exaggerate conclusions.
    7. If information is unavailable, write:
    "Not explicitly stated in the paper."
    8. Clearly distinguish between:
    - What the authors did
    - What the authors found
    - What the authors concluded

    Create the summary using EXACTLY these headings:
    1.Research Problem
    2.Motivation
    3.Methodology
    4.Dataset / Experimental Setup
    5.Key Results
    6.Limitations
    7.Plain Language Explanation

    RESEARCH PAPER:{text}
    """
    return call_gemini(prompt)

def summarize_chunk(chunk):
    prompt = f"""
    You are a research paper analysis system.
    Summarize this section faithfully.
    RULES:
    - Use ONLY information present in this text.
    - Do NOT invent facts.
    - Preserve numerical values exactly.
    - Do NOT infer missing details.
    - Focus on research problem, methodology,
    experiments, results and limitations.
    TEXT:{chunk}"""
    return call_gemini(prompt)

def generate_final_summary(intermediate_summaries):
    combined_summary = "\n\n".join(intermediate_summaries)

    prompt = f"""
    You are ResearchLens AI.
    Using ONLY the intermediate summaries below, create one accurate structured summary.
    Do NOT add information that is not present.
    Use EXACTLY these headings:
    1.Research Problem
    2.Motivation
    3.Methodology
    4.Dataset / Experimental Setup
    5.Key Results
    6.Limitations
    7.Plain Language Explanation
    
    STRICT RULES:
    - Do NOT invent information.
    - Preserve numerical results exactly.
    - Do NOT combine different experiments incorrectly.
    - If information is missing, write:
    "Not explicitly stated in the paper."
    INTERMEDIATE SUMMARIES: {combined_summary}
    """
    return call_gemini(prompt)

def build_glossary(paper_text):
    """
    Extract domain-specific terms, acronyms, and jargon from the paper
    and return plain-language definitions + category tags.
    """
    prompt = f"""
You are an expert research paper analyst and science communicator.
Read the research paper below and extract the most important technical terms,
acronyms, and domain-specific concepts that a reader MUST understand to
follow the paper.

Return ONLY valid JSON — an array of objects with this EXACT structure:
[
  {{
    "term": "Exact term or acronym as used in the paper",
    "full_form": "Full form if it is an acronym, otherwise null",
    "definition": "Clear 1–2 sentence plain-English definition based ONLY on the paper",
    "category": "One of: Model/Architecture | Dataset | Metric | Method | Concept | Tool | Other",
    "first_used": "Brief quote or context showing where the term first appears (max 20 words)"
  }}
]

Rules:
- Extract 10–25 of the most important terms (prioritise acronyms, model names, metrics, novel methods).
- Definitions must be grounded in the paper — do NOT invent information.
- If a term's meaning is not explained in the paper, write "Defined implicitly; standard term in the field."
- Return ONLY the JSON array, no extra text, no markdown fences.

RESEARCH PAPER:
{paper_text[:20000]}
"""
    response = call_gemini(prompt)
    try:
        response = response.replace("```json", "").replace("```", "").strip()
        start = response.find("[")
        end   = response.rfind("]") + 1
        if start != -1 and end > start:
            return json.loads(response[start:end])
        return []
    except Exception:
        return []


def extract_citations(paper_text):
    """
    Extract and format the references / citations section from the paper.
    Returns a list of citation dicts with index, text, and optional DOI/URL.
    """
    prompt = f"""
You are a research paper citation extractor.
Extract ALL references listed in the REFERENCES or BIBLIOGRAPHY section of the paper below.

Return ONLY valid JSON — an array of objects with this exact structure:
[
  {{
    "index": 1,
    "authors": "Author names as written in the paper",
    "title": "Title of the referenced paper/book",
    "venue": "Journal, conference, or publisher name (if present)",
    "year": "Year of publication (if present)",
    "doi_or_url": "DOI or URL if present, otherwise null"
  }}
]

Rules:
- Include every reference you find.
- Do NOT invent information not present in the paper.
- If a field is missing, use null.
- Return ONLY the JSON array, no extra text.

RESEARCH PAPER:
{paper_text[-15000:]}
"""
    response = call_gemini(prompt)
    try:
        response = response.replace("```json", "").replace("```", "").strip()
        # Find the JSON array
        start = response.find("[")
        end = response.rfind("]") + 1
        if start != -1 and end > start:
            return json.loads(response[start:end])
        return []
    except Exception:
        return []


def extract_claims(summary):
    lines = summary.split("\n")
    claims = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if len(line) < 25:
            continue
        claims.append(line)
    return claims[:8]

def verify_claim(claim, paper_text):
    prompt = f"""
    You are a research paper fact-checking system.
    Check whether the following CLAIM is supported by the RESEARCH PAPER.
    CLAIM:{claim}
    RESEARCH PAPER: {paper_text}
    Return ONLY valid JSON.
    Use this exact structure:
    {{
    "verdict": "SUPPORTED",
    "reason": "Short explanation",
    "evidence": "Exact or closely matching evidence from the paper"
    }}
    Possible verdicts:
    SUPPORTED
    PARTIALLY_SUPPORTED
    NOT_SUPPORTED
    UNCLEAR
    Rules:
    - Use only the research paper.
    - Do not invent evidence.
    - Numerical values must match exactly.
    - If the paper does not clearly support the claim, do not mark it SUPPORTED.
    """
    response = call_gemini(prompt)
    try:
        response = response.replace(
            "```json",
            ""
        ).replace(
            "```",
            ""
        ).strip()
        return json.loads(response)
    except Exception:
        return {
            "verdict": "UNCLEAR",
            "reason":
            "The verification response could not be parsed.",
            "evidence":
            "Verification output unavailable."
        }

st.markdown("""
    <div class="research-header">
        <div class="research-badge">
            AI-POWERED RESEARCH ANALYSIS
        </div>
        <div class="research-title">
            ResearchLens AI
        </div>
        <div class="research-subtitle">
            Faithful Research Paper Summarization
            with Evidence-Based Verification
        </div>
    </div>
    """,unsafe_allow_html=True )

st.markdown("""
    <div class="intro-card">
        <div class="intro-text">
        Upload a research paper and ResearchLens AI will
        extract its content, analyze the research methodology
        and results, generate a structured plain-language
        summary, and verify important claims against the
        original paper.
        </div>
    </div>
    """,unsafe_allow_html=True)

st.markdown("""
    <div class="section-title">Upload Research Paper</div>
    """, unsafe_allow_html=True)

st.markdown("""
    <div class="upload-description">
        Upload a PDF research paper to begin analysis.
    </div>
    """,unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    try:
        paper_text = extract_text_from_pdf(uploaded_file)
        paper_text = clean_text(paper_text)
    except Exception as e:
        st.error(f"Could not read PDF: {e}")
        st.stop()
    if not paper_text:
        st.error("No readable text was found in this PDF.")
        st.stop()

    estimated_words = len(paper_text.split())
    estimated_pages = max(1, len(paper_text) // 3000)

    st.markdown("""
        <div class="section-title">
            Paper Overview
        </div>
        """,unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Characters Extracted",
            f"{len(paper_text):,}"
        )
    with col2:
        st.metric(
            "Estimated Words",
            f"{estimated_words:,}"
        )
    with col3:
        st.metric(
            "Estimated Pages",
            estimated_pages
        )

    st.markdown(f"""
        <div class="file-info">
            📄 {uploaded_file.name}
        </div>
        """,unsafe_allow_html=True)

    with st.expander("View Extracted Paper Text"):
        st.text(paper_text[:8000])

    st.markdown(
        '<div class="section-title">AI Analysis</div>',
        unsafe_allow_html=True
    )
    generate_button = st.button(
        "Generate Research Summary",
        use_container_width=True
    )

    if generate_button:
        progress_bar = st.progress(0)
        status_text = st.empty()

        if len(paper_text) <= 25000:
            status_text.info("Analyzing research paper...")
            try:
                final_summary = summarize_paper(paper_text)
            except Exception as e:
                st.error(
                    f"Error generating summary: {e}"
                )
                st.stop()
            progress_bar.progress(100)

        else:
            chunks = chunk_text(paper_text)
            intermediate_summaries = []
            for i, chunk in enumerate(chunks):
                status_text.info(
                    f"Analyzing section {i + 1} "
                    f"of {len(chunks)}..."
                )
                try:
                    summary = summarize_chunk(
                        chunk
                    )
                    intermediate_summaries.append(
                        summary
                    )
                except Exception as e:
                    st.error(
                        f"Error analyzing section "
                        f"{i + 1}: {e}"
                    )
                    st.stop()

                progress = int(
                    ((i + 1) / len(chunks))
                    * 80
                )
                progress_bar.progress(
                    progress
                )
                if i < len(chunks) - 1:
                    time.sleep(4)

            status_text.info(
                "Generating final structured summary..."
            )
            try:
                final_summary = (
                    generate_final_summary(
                        intermediate_summaries
                    )
                )
            except Exception as e:
                st.error(
                    f"Error generating final summary: {e}"
                )
                st.stop()
            progress_bar.progress(100)

        status_text.success("Research paper analysis completed!")
        
        st.session_state["final_summary"] = final_summary

        st.session_state["paper_text"] = paper_text

if "final_summary" in st.session_state:
    final_summary = st.session_state[
        "final_summary"
    ]
    paper_text = st.session_state[
        "paper_text"
    ]
    st.markdown("---")

    st.markdown(
        '<div class="summary-heading">'
        'Research Summary'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        final_summary
    )

    st.markdown("---")

    st.markdown("""
        <div class="section-title">
            Faithfulness Check
        </div>
        """,unsafe_allow_html=True)

    st.write(
        "Important claims from the generated summary "
        "can be verified against the original research paper."
    )

    if st.button(
        "Verify Summary Claims",
        use_container_width=True
    ):
        claims = extract_claims(
            final_summary
        )
        if not claims:
            st.warning(
                "No suitable claims were found."
            )
        else:
            verification_progress = st.progress(0)
            verified_claims = []
            for i, claim in enumerate(claims):
                try:
                    result = verify_claim(
                        claim,
                        paper_text
                    )
                    verified_claims.append(
                        {
                            "claim": claim,
                            "result": result
                        }
                    )
                except Exception as e:
                    verified_claims.append(
                        {
                            "claim": claim,
                            "result": {
                                "verdict":
                                "UNCLEAR",
                                "reason":
                                str(e),
                                "evidence":
                                "Verification unavailable."
                            }
                        }
                    )
                progress = int(
                    ((i + 1) / len(claims))
                    * 100
                )
                verification_progress.progress(
                    progress
                )
                if i < len(claims) - 1:
                    time.sleep(4)

            st.session_state[
                "verified_claims"
            ] = verified_claims

if "verified_claims" in st.session_state:
    verified_claims = st.session_state[
        "verified_claims"
    ]
    supported_count = sum(
        1 for item in verified_claims
        if item["result"]
        .get("verdict")
        == "SUPPORTED"
    )
    total_claims = len(
        verified_claims
    )
    faithfulness_score = round(
        (
            supported_count
            / total_claims
        ) * 100
        if total_claims > 0
        else 0
    )
    st.markdown("""
        <div class="faithfulness-score-card">
        """,unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Claims Verified",
            f"{supported_count} / {total_claims}"
        )
    with col2:
        st.metric(
            "Faithfulness Score",
            f"{faithfulness_score}%"
        )
    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    for index, item in enumerate(verified_claims, start=1):
        claim = item["claim"]
        result = item["result"]
        verdict = result.get(
            "verdict",
            "UNCLEAR"
        )
        reason = result.get(
            "reason",
            "No explanation available."
        )
        evidence = result.get(
            "evidence",
            "No evidence available."
        )
        verdict_class = verdict.lower().replace(
            "_",
            "-"
        )
        st.markdown(f"""
            <div class="claim-card">
                <div class="claim-id">
                    Claim {index}
                </div>
                <div class="claim-text">
                    {claim}
                </div>
                <div class="verification-status
                {verdict_class}">{verdict}
                </div>
                <div class="verification-reason">
                    {reason}
                </div>
                <div class="evidence-box">
                    <b>Evidence from Paper:</b>
                    <br><br>
                    {evidence}
                </div>
            </div>
            """,unsafe_allow_html=True)



if st.session_state.get("paper_text"):

    # ── Technical Glossary ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-title">🧠 Technical Glossary</div>',
        unsafe_allow_html=True
    )
    st.write(
        "Instantly decode every technical term, acronym, and piece of jargon "
        "in the paper — with plain-English definitions grounded in the text."
    )

    if st.button("Build Glossary", use_container_width=True, key="build_glossary_btn"):
        with st.spinner("Extracting and defining key terms…"):
            glossary = build_glossary(st.session_state.paper_text)
        if not glossary:
            st.warning("Could not extract terms. Try again or check the paper text.")
        else:
            st.session_state["glossary"] = glossary
            st.success(f"Defined **{len(glossary)}** key terms from the paper.")

    if st.session_state.get("glossary"):
        glossary = st.session_state["glossary"]

        # ── Category filter + search ─────────────────────────────────────
        all_categories = sorted({
            str(t.get("category") or "Other") for t in glossary
        })
        col_search, col_filter = st.columns([3, 2])
        with col_search:
            g_search = st.text_input(
                "Search terms",
                placeholder="Search term or keyword…",
                label_visibility="collapsed",
                key="glossary_search"
            )
        with col_filter:
            g_category = st.selectbox(
                "Filter by category",
                options=["All"] + all_categories,
                label_visibility="collapsed",
                key="glossary_cat"
            )

        filtered_terms = glossary
        if g_search:
            q = g_search.lower()
            filtered_terms = [
                t for t in filtered_terms
                if q in str(t.get("term") or "").lower()
                or q in str(t.get("definition") or "").lower()
                or q in str(t.get("full_form") or "").lower()
            ]
        if g_category != "All":
            filtered_terms = [
                t for t in filtered_terms
                if str(t.get("category") or "Other") == g_category
            ]

        # Category pill counts
        cat_counts = {}
        for t in glossary:
            c = str(t.get("category") or "Other")
            cat_counts[c] = cat_counts.get(c, 0) + 1

        pills_html = " ".join(
            f'<span class="glossary-pill">{html.escape(cat)} <span class="pill-count">{cnt}</span></span>'
            for cat, cnt in sorted(cat_counts.items())
        )
        st.markdown(
            f'<div class="glossary-pills">{pills_html}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="glossary-count">'
            f'Showing {len(filtered_terms)} of {len(glossary)} terms'
            f'</div>',
            unsafe_allow_html=True
        )

        # Render term cards
        for term_obj in filtered_terms:
            term       = html.escape(str(term_obj.get("term") or ""))
            full_form  = html.escape(str(term_obj.get("full_form") or ""))
            definition = html.escape(str(term_obj.get("definition") or ""))
            category   = html.escape(str(term_obj.get("category") or "Other"))
            first_used = html.escape(str(term_obj.get("first_used") or ""))

            full_form_html = (
                f'<span class="glossary-full-form">— {full_form}</span>'
                if full_form else ""
            )
            first_used_html = (
                f'<div class="glossary-first-used">'
                f'<span class="glossary-first-used-label">In paper: </span>'
                f'"{first_used}"'
                f'</div>'
                if first_used else ""
            )

            cat_slug = re.sub(r"[^a-z0-9-]", "-", category.lower())

            card_html = textwrap.dedent(f"""\
                <div class="glossary-card">
                <div class="glossary-card-header">
                <div class="glossary-term-group">
                <span class="glossary-term">{term}</span>
                {full_form_html}
                </div>
                <span class="glossary-category {cat_slug}">{category}</span>
                </div>
                <div class="glossary-definition">{definition}</div>
                {first_used_html}
                </div>""")
            st.markdown(card_html, unsafe_allow_html=True)

        # Download glossary as JSON
        glossary_json = json.dumps(glossary, indent=2, ensure_ascii=False)
        st.download_button(
            label="⬇️ Download Glossary as JSON",
            data=glossary_json,
            file_name="glossary.json",
            mime="application/json",
            use_container_width=True,
            key="download_glossary_btn"
        )

    # ── Citation Extractor ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-title">📚 Citation Extractor</div>',
        unsafe_allow_html=True
    )
    st.write(
        "Extract and browse all references cited in the paper. "
        "Each citation is parsed into authors, title, venue, year, and link."
    )

    if st.button("Extract Citations", use_container_width=True, key="extract_citations_btn"):
        with st.spinner("Extracting citations from the paper…"):
            citations = extract_citations(st.session_state.paper_text)
        if not citations:
            st.warning(
                "No citations could be extracted. "
                "The paper may not have a structured reference list."
            )
        else:
            st.session_state["citations"] = citations
            st.success(f"Found **{len(citations)}** references.")

    if st.session_state.get("citations"):
        citations = st.session_state["citations"]

        # Search / filter bar
        search_term = st.text_input(
            "🔍 Filter citations",
            placeholder="Type an author name, keyword, or year…",
            label_visibility="collapsed",
            key="citations_search"
        )

        filtered = citations
        if search_term:
            q = search_term.lower()
            filtered = [
                c for c in citations
                if q in str(c.get("authors", "")).lower()
                or q in str(c.get("title", "")).lower()
                or q in str(c.get("venue", "")).lower()
                or q in str(c.get("year", "")).lower()
            ]

        st.markdown(
            f'<div class="citation-count">'
            f'Showing {len(filtered)} of {len(citations)} references'
            f'</div>',
            unsafe_allow_html=True
        )

        for cite in filtered:
            idx        = html.escape(str(cite.get("index", "")))
            authors    = html.escape(str(cite.get("authors") or "Unknown authors"))
            title      = html.escape(str(cite.get("title") or "Untitled"))
            venue      = html.escape(str(cite.get("venue") or ""))
            year       = html.escape(str(cite.get("year") or ""))
            doi_or_url = cite.get("doi_or_url")

            venue_year = ", ".join(filter(None, [venue, year]))

            link_html = ""
            if doi_or_url:
                raw_url = str(doi_or_url).strip()
                href = html.escape(
                    raw_url if raw_url.startswith("http") else f"https://doi.org/{raw_url}"
                )
                link_html = (
                    f'<a class="citation-link" '
                    f'href="{href}" target="_blank" rel="noopener">'
                    f'🔗 View source</a>'
                )

            cite_html = textwrap.dedent(f"""\
                <div class="citation-card">
                <div class="citation-index">Ref {idx}</div>
                <div class="citation-title">{title}</div>
                <div class="citation-authors">{authors}</div>
                {'<div class="citation-venue">' + venue_year + '</div>' if venue_year else ''}
                {link_html}
                </div>""")
            st.markdown(cite_html, unsafe_allow_html=True)

        # Download all citations as JSON
        citations_json = json.dumps(citations, indent=2, ensure_ascii=False)
        st.download_button(
            label="⬇️ Download Citations as JSON",
            data=citations_json,
            file_name="citations.json",
            mime="application/json",
            use_container_width=True,
            key="download_citations_btn"
        )

    # ── Ask the Paper ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💬 Ask the Paper")
    st.write(
        "Ask questions about the uploaded research paper. "
        "Answers are generated using relevant text from the paper."
    )

    # Keep the conversation across Streamlit reruns
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    # Display previous questions and answers
    for message in st.session_state.qa_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input box for a new question
    question = st.chat_input(
        "Ask something about your research paper..."
    )

    if question:
        # Display and save the user's question
        st.session_state.qa_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        # Generate an answer from the paper
        with st.chat_message("assistant"):
            with st.spinner("Finding relevant information..."):
                answer = answer_question(
                    question,
                    st.session_state.paper_text
                )

            st.markdown(answer)

        # Save the answer so it remains visible
        st.session_state.qa_history.append({
            "role": "assistant",
            "content": answer
        })

st.markdown("""
    <div class="footer">
        ResearchLens AI · Faithful Research Paper Summarization
    </div>
    """,unsafe_allow_html=True)