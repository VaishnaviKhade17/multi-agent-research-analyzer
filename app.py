"""
Streamlit UI for the multi-agent research pipeline defined in pipeline.py

Run with:
    streamlit run streamlit_app.py

Place this file in the same folder as pipeline.py, agents.py, tools.py, .env
"""

import io
import queue
import sys
import threading
import time
from datetime import datetime

import streamlit as st

from pipeline import run_research_pipeline


# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🔎",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem;}
        .stTextArea textarea {font-family: "Source Code Pro", monospace; font-size: 0.85rem;}
        .log-box {
            background-color: #0e1117;
            color: #d1d5db;
            padding: 1rem;
            border-radius: 8px;
            font-family: "Source Code Pro", monospace;
            font-size: 0.8rem;
            white-space: pre-wrap;
            max-height: 420px;
            overflow-y: auto;
            border: 1px solid #262730;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Helpers: run the pipeline in a background thread and stream its stdout
# --------------------------------------------------------------------------
class QueueWriter(io.TextIOBase):
    """A file-like object that pushes every write() into a thread-safe queue."""

    def __init__(self, q: "queue.Queue[str]"):
        self.q = q

    def write(self, text):
        if text:
            self.q.put(text)
        return len(text)

    def flush(self):
        pass


def _pipeline_worker(topic: str, log_queue: "queue.Queue[str]", result_holder: dict):
    """Runs in a background thread: redirects stdout to the queue, runs the
    pipeline, and stores either the result or the error in result_holder."""
    old_stdout = sys.stdout
    sys.stdout = QueueWriter(log_queue)
    try:
        result = run_research_pipeline(topic)
        result_holder["result"] = result
    except Exception as exc:  # noqa: BLE001
        result_holder["error"] = exc
    finally:
        sys.stdout = old_stdout
        result_holder["done"] = True


# --------------------------------------------------------------------------
# Session state init
# --------------------------------------------------------------------------
if "log_text" not in st.session_state:
    st.session_state.log_text = ""
if "running" not in st.session_state:
    st.session_state.running = False
if "result" not in st.session_state:
    st.session_state.result = None
if "error" not in st.session_state:
    st.session_state.error = None
if "history" not in st.session_state:
    st.session_state.history = []  # list of (timestamp, topic)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("🔎 Research Pipeline")
    st.caption("Search agent → Reader agent → Writer chain → Critic chain")
    st.divider()
    st.markdown(
        "**Pipeline stages**\n\n"
        "1. **Search agent** — finds recent, reliable sources\n"
        "2. **Reader agent** — scrapes the most relevant URL\n"
        "3. **Writer chain** — drafts the report\n"
        "4. **Critic chain** — reviews the draft"
    )
    st.divider()
    if st.session_state.history:
        st.subheader("Recent topics")
        for ts, t in reversed(st.session_state.history[-8:]):
            st.caption(f"{ts} — {t}")


# --------------------------------------------------------------------------
# Main input
# --------------------------------------------------------------------------
st.title("Multi-Agent Research System")

with st.form("topic_form", clear_on_submit=False):
    topic = st.text_input(
        "Research topic",
        placeholder="e.g. Recent advances in solid-state batteries",
        disabled=st.session_state.running,
    )
    submitted = st.form_submit_button(
        "Run pipeline", disabled=st.session_state.running, use_container_width=True
    )

if submitted:
    if not topic or not topic.strip():
        st.warning("Please enter a topic before running the pipeline.")
    else:
        st.session_state.running = True
        st.session_state.log_text = ""
        st.session_state.result = None
        st.session_state.error = None
        st.session_state.history.append((datetime.now().strftime("%H:%M:%S"), topic.strip()))

        log_queue: "queue.Queue[str]" = queue.Queue()
        result_holder: dict = {"done": False}

        thread = threading.Thread(
            target=_pipeline_worker, args=(topic.strip(), log_queue, result_holder), daemon=True
        )
        thread.start()

        status_placeholder = st.empty()
        log_placeholder = st.empty()

        with status_placeholder.container():
            st.info("Pipeline running — this can take a minute or two depending on the sources found.")

        while not result_holder.get("done"):
            drained = False
            while not log_queue.empty():
                st.session_state.log_text += log_queue.get()
                drained = True
            if drained:
                log_placeholder.markdown(
                    f'<div class="log-box">{st.session_state.log_text}</div>',
                    unsafe_allow_html=True,
                )
            time.sleep(0.2)

        # Drain anything left after completion
        while not log_queue.empty():
            st.session_state.log_text += log_queue.get()
        log_placeholder.markdown(
            f'<div class="log-box">{st.session_state.log_text}</div>', unsafe_allow_html=True
        )

        st.session_state.running = False
        st.session_state.result = result_holder.get("result")
        st.session_state.error = result_holder.get("error")
        status_placeholder.empty()
        st.rerun()


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
if st.session_state.error is not None:
    st.error(f"Pipeline failed: {st.session_state.error}")
    if st.session_state.log_text:
        with st.expander("Full log", expanded=True):
            st.markdown(
                f'<div class="log-box">{st.session_state.log_text}</div>', unsafe_allow_html=True
            )

elif st.session_state.result is not None:
    result = st.session_state.result
    st.success("Pipeline completed.")

    tab_report, tab_critic, tab_search, tab_scraped, tab_log = st.tabs(
        ["📄 Report", "🧐 Critic Feedback", "🔍 Search Results", "📰 Scraped Content", "🧾 Full Log"]
    )

    with tab_report:
        st.markdown(result.get("report", "_No report generated._"))
        st.download_button(
            "Download report as .md",
            data=str(result.get("report", "")),
            file_name="research_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with tab_critic:
        st.markdown(result.get("feedback", "_No feedback generated._"))

    with tab_search:
        st.text(result.get("search_results", "_No search results._"))

    with tab_scraped:
        st.text(result.get("scraped_content", "_No scraped content._"))

    with tab_log:
        st.markdown(
            f'<div class="log-box">{st.session_state.log_text}</div>', unsafe_allow_html=True
        )

else:
    st.caption("Enter a topic above and click **Run pipeline** to start.")