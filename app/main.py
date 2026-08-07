"""
app/main.py

Full Streamlit UI for the Offline Multimodal RAG system.
Run with: streamlit run app/main.py
"""

import sys
import tempfile
import os
import shutil
from pathlib import Path

import wave
import numpy as np

import httpx
import ollama
import streamlit as st
from streamlit_mic_recorder import mic_recorder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, SUMMARY_TOP_K
from ingestion.ingest import ingest_file, validate_file
from ingestion.chunker import chunk_text
from ingestion.audio_extractor import extract_audio
from retrieval.vector_store import VectorStore
from app.rag_pipeline import generate_answer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACCEPTED_EXTENSIONS = ["pdf", "docx", "png", "jpg", "jpeg", "wav", "mp3"]

_EXT_LABELS = {
    "pdf": "PDF", "docx": "Word", "png": "Image",
    "jpg": "Image", "jpeg": "Image", "wav": "Audio", "mp3": "Audio",
}

_SUMMARY_KEYWORDS = {"summarize", "summary", "summarise", "overview", "outline"}

def _is_summary_query(query: str) -> bool:
    return bool(_SUMMARY_KEYWORDS & set(query.lower().split()))


# ---------------------------------------------------------------------------
# Cached singletons — created once per Streamlit session
# ---------------------------------------------------------------------------

@st.cache_resource
def get_store() -> VectorStore:
    return VectorStore()


def _list_ollama_models() -> list[str]:
    """Return locally available Ollama model names, or [] on connection error."""
    try:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        return [m["model"] for m in client.list().get("models", [])]
    except (httpx.ConnectError, Exception):
        return []


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "chat_history":        [],    # list of {query, answer, sources}
        "confirm_reset":       False, # two-step reset flag
        "active_model":        OLLAMA_MODEL,
        "pending_voice_query": "",    # transcribed text awaiting confirmation
        "last_audio_bytes":    None,  # dedupe: skip re-transcribing same clip
        "scoped_source":       None,  # currently selected document scope
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# Ingestion helper — granular per-stage progress
# ---------------------------------------------------------------------------

def _ingest_uploaded_file(uploaded_file, store: VectorStore) -> tuple[int, str, str]:
    """
    Save → validate → extract → chunk → embed → store.

    Returns:
        (chunks_added, error_message, warning_message)
        Exactly one of the three values will be non-zero/non-empty.
    """
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        status = st.status(f"Processing **{uploaded_file.name}**…", expanded=True)

        with status:
            # Stage 1 — validate
            st.write("🔍 Validating file…")
            try:
                validate_file(tmp_path)
            except (FileNotFoundError, ValueError) as exc:
                raise ValueError(
                    str(exc).replace(Path(tmp_path).name, uploaded_file.name)
                ) from exc

            # Stage 2 — extract text
            st.write("📄 Extracting text…")
            pages = ingest_file(tmp_path)
            if not pages:
                raise ValueError(
                    f"No text could be extracted from **{uploaded_file.name}**. "
                    "Check that the file is not blank, image-only without OCR support, "
                    "or in an unsupported format."
                )

            # Stage 3 — chunk
            st.write(f"✂️ Chunking into segments… ({len(pages)} page/block(s) found)")
            all_chunks: list[dict] = []
            for page in pages:
                meta = {**page["metadata"], "source": uploaded_file.name}
                all_chunks.extend(chunk_text(page["text"], meta))

            # Stage 4 — embed & store
            st.write(f"🧠 Embedding and storing {len(all_chunks)} chunk(s)…")
            added = store.store_chunks(all_chunks)

        if added == 0:
            status.update(label=f"⚠️ **{uploaded_file.name}** — already in knowledge base", state="complete")
            return 0, "", (
                f"**{uploaded_file.name}** is already in the knowledge base — "
                "no new chunks were added. To re-ingest with different settings, "
                "reset the knowledge base first."
            )

        status.update(
            label=f"✅ **{uploaded_file.name}** — {added} chunk(s) stored",
            state="complete",
        )
        return added, "", ""

    except ValueError as exc:
        print(f"[ingest] ValueError: {exc}")
        return 0, str(exc), ""
    except Exception as exc:
        print(f"[ingest] {type(exc).__name__}: {exc}")
        return 0, (
            f"Unexpected error while processing **{uploaded_file.name}**: {exc}. "
            "Try re-uploading the file or check the terminal for details."
        ), ""
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar(store: VectorStore) -> None:
    with st.sidebar:
        st.title("⚙️ Knowledge Base")

        # --- ffmpeg check ---------------------------------------------------
        if shutil.which("ffmpeg") is None:
            st.warning(
                "⚠️ **ffmpeg not found on PATH.** Voice recording may fail to decode.\n\n"
                "**Fix:** Install ffmpeg and ensure it is on your system PATH, "
                "then restart the app."
            )
            st.divider()

        # --- Stats ----------------------------------------------------------
        chunk_count = store.count()
        sources     = store.list_ingested_sources()
        st.metric("Chunks stored", chunk_count)
        st.metric("Files ingested", len(sources))
        st.divider()

        # --- Model selector -------------------------------------------------
        st.subheader("🤖 Model")
        models = _list_ollama_models()
        if models:
            current     = st.session_state.active_model
            default_idx = models.index(current) if current in models else 0
            st.session_state.active_model = st.selectbox(
                "Ollama model", models, index=default_idx, label_visibility="collapsed"
            )
        else:
            st.warning(
                "**Ollama is not reachable.** Answers won't be generated until it's running.\n\n"
                "**Fix:** Open a terminal and run `ollama serve`, then refresh this page."
            )
            st.caption(f"Configured model: `{OLLAMA_MODEL}`")
        st.divider()

        # --- Ingested files -------------------------------------------------
        st.subheader("📋 Ingested Files")
        if not sources:
            st.info("No documents ingested yet.", icon="ℹ️")
        else:
            for entry in sources:
                col_f, col_r = st.columns([5, 1])
                ext   = Path(entry["source"]).suffix.lstrip(".")
                label = _EXT_LABELS.get(ext, ext.upper())
                col_f.caption(f"`{entry['source']}` · {label} · {entry['chunk_count']} chunks")
                if col_r.button("🗑", key=f"rm_{entry['source']}", help=f"Remove {entry['source']}"):
                    store.remove_source(entry["source"])
                    st.rerun()
        st.divider()

        # --- Reset KB -------------------------------------------------------
        st.subheader("🗑️ Reset Knowledge Base")
        if not st.session_state.confirm_reset:
            if st.button("Reset Knowledge Base", type="secondary", use_container_width=True):
                st.session_state.confirm_reset = True
                st.rerun()
        else:
            st.warning(
                "⚠️ This will permanently delete **all** stored chunks and clear "
                "the chat history. This cannot be undone."
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Confirm", type="primary", use_container_width=True):
                    with st.spinner("Clearing knowledge base…"):
                        store.reset()
                    st.session_state.chat_history = []
                    st.session_state.confirm_reset = False
                    st.success("Knowledge base cleared. Upload new documents to get started.")
                    st.rerun()
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.confirm_reset = False
                    st.rerun()


# ---------------------------------------------------------------------------
# Upload panel
# ---------------------------------------------------------------------------

def _render_upload_panel(store: VectorStore) -> None:
    st.header("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload files to add to the knowledge base",
        type=ACCEPTED_EXTENSIONS,
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        ingested_sources = {s["source"] for s in store.list_ingested_sources()}
        new_files = [f for f in uploaded_files if f.name not in ingested_sources]
        for uf in new_files:
            added, err, warn = _ingest_uploaded_file(uf, store)
            if err:
                st.error(
                    f"❌ **Could not ingest {uf.name}**\n\n{err}",
                    icon="🚨",
                )
            elif warn:
                st.warning(f"⚠️ {warn}", icon="⚠️")


# ---------------------------------------------------------------------------
# Shared query execution — single path for typed + voice
# ---------------------------------------------------------------------------

def _submit_query(query: str, store: VectorStore, source: str | None = None) -> None:
    """Search the KB and generate an answer; append result to chat history."""
    if store.count() == 0:
        st.session_state.chat_history.append({
            "query":   query,
            "answer":  (
                "⚠️ The knowledge base is empty. "
                "Upload at least one document before asking questions."
            ),
            "sources": [],
        })
        return

    top_k = SUMMARY_TOP_K if (source and _is_summary_query(query)) else None
    scope_label = f" in **{source}**" if source else ""

    # Stage 1 — retrieval
    with st.spinner(f"🔎 Searching knowledge base{scope_label}…"):
        chunks = store.search(query, top_k=top_k, source=source)

    # Stage 2 — generation
    with st.spinner(f"💬 Generating answer with **{st.session_state.active_model}**…"):
        result = generate_answer(
            query,
            chunks,
            model=st.session_state.active_model,
        )

    st.session_state.chat_history.append({
        "query":   query,
        "answer":  result["answer"],
        "sources": result["sources"],
    })


# ---------------------------------------------------------------------------
# Voice transcription helper
# ---------------------------------------------------------------------------

def _transcribe_recording(audio: dict) -> str:
    """
    Write mic_recorder output to a proper WAV file and transcribe with Whisper.

    Returns transcribed text, or "" on failure (error shown inline).
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = tmp.name

    try:
        with open(tmp_path, "wb") as f:
            f.write(audio["bytes"])

        # --- Validate the written WAV ---
        with wave.open(tmp_path, "rb") as wf:
            n_frames     = wf.getnframes()
            framerate    = wf.getframerate()
            sample_width = wf.getsampwidth()
            raw          = wf.readframes(n_frames)

        duration = n_frames / framerate if framerate else 0
        if duration < 0.1 or len(raw) < 2:
            raise ValueError("Recording appears to be empty or too short.")

        dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sample_width, np.int16)
        samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)
        rms = np.sqrt(np.mean(samples ** 2)) if samples.size else 0
        if rms < 1.0:
            raise ValueError("Recording appears to be silent or corrupted.")

        result = extract_audio(tmp_path)
        return result["text"].strip()

    except FileNotFoundError as exc:
        print(f"[transcribe] FileNotFoundError: {exc}")
        st.error(
            "❌ **Transcription failed:** the recorded audio could not be saved. "
            "Try recording again.",
            icon="🚨",
        )
        return ""
    except ValueError as exc:
        print(f"[transcribe] ValueError: {exc}")
        st.error(
            f"❌ **Transcription failed:** {exc} "
            "Try speaking more clearly or moving to a quieter environment.",
            icon="🚨",
        )
        return ""
    except Exception as exc:
        print(f"[transcribe] {type(exc).__name__}: {exc}")
        st.error(
            f"❌ **Transcription failed:** {exc} "
            "Check that the Whisper model has been downloaded (`./models/whisper`).",
            icon="🚨",
        )
        return ""
    finally:
        # DEBUG: copy recording for manual inspection before deleting
        debug_path = Path(__file__).resolve().parents[1] / "debug_last_recording.wav"
        shutil.copy2(tmp_path, debug_path)
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Ask + Answer panels
# ---------------------------------------------------------------------------

def _render_ask_panel(store: VectorStore) -> None:
    st.header("💬 Ask a Question")

    # ── Scope selector (outside form so it updates immediately) ──────────
    sources = store.list_ingested_sources()
    source_options = ["All documents"] + [s["source"] for s in sources]
    selected = st.selectbox(
        "🔍 Scope search to document:",
        source_options,
        key="scope_selectbox",
    )
    st.session_state.scoped_source = None if selected == "All documents" else selected

    # ── Typed input ────────────────────────────────────────────────────────
    with st.form("ask_form", clear_on_submit=True):
        query = st.text_input(
            "Your question",
            placeholder="e.g. What are the main findings in the report?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)

    if submitted:
        if not query.strip():
            st.warning(
                "⚠️ Please type a question before submitting.",
                icon="⚠️",
            )
        else:
            _submit_query(query.strip(), store, source=st.session_state.scoped_source)

    # ── Voice input ────────────────────────────────────────────────────────
    st.markdown("**Or record a voice question:**")
    audio = mic_recorder(
        start_prompt="🎙️ Start recording",
        stop_prompt="⏹️ Stop recording",
        just_once=True,
        use_container_width=True,
        format="wav",
        key="mic",
    )

    if audio and audio["bytes"] != st.session_state.last_audio_bytes:
        st.session_state.last_audio_bytes = audio["bytes"]
        with st.spinner("🎧 Transcribing your recording…"):
            transcript = _transcribe_recording(audio)
        if transcript:
            st.session_state.pending_voice_query = transcript
        else:
            st.warning(
                "⚠️ **No speech detected.** The recording may be too short or too quiet. "
                "Try again in a quieter environment and speak clearly into the microphone.",
                icon="⚠️",
            )

    # ── Confirm / edit step for voice ──────────────────────────────────────
    if st.session_state.pending_voice_query:
        st.info("🎤 **Transcribed question** — edit if needed, then submit:", icon="ℹ️")
        edited = st.text_area(
            "Transcribed question",
            value=st.session_state.pending_voice_query,
            height=80,
            label_visibility="collapsed",
            key="voice_edit",
        )
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Submit", type="primary", use_container_width=True, key="voice_submit"):
                st.session_state.pending_voice_query = ""
                if edited.strip():
                    _submit_query(edited.strip(), store, source=st.session_state.scoped_source)
                else:
                    st.warning(
                        "⚠️ The transcribed question is empty. "
                        "Record again or type your question above.",
                        icon="⚠️",
                    )
        with col2:
            if st.button("Discard", use_container_width=True, key="voice_discard"):
                st.session_state.pending_voice_query = ""
                st.rerun()

    _render_answer_panel()


def _render_answer_panel() -> None:
    st.divider()

    if not st.session_state.chat_history:
        st.info(
            "💡 **Ask a question to get started.**\n\n"
            "Type in the box above or record a voice question. "
            "Make sure you've uploaded at least one document first.",
            icon="ℹ️",
        )
        return

    for turn in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(turn["query"])

        with st.chat_message("assistant"):
            st.markdown(turn["answer"])

            if turn["sources"]:
                with st.expander(f"📎 Sources ({len(turn['sources'])})"):
                    for src in turn["sources"]:
                        st.markdown(
                            f"**{src['source']}** — *{src['page_or_timestamp']}*"
                        )
                        st.caption(f"> {src['snippet']}")
                        st.divider()


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Offline RAG Assistant",
        page_icon="🧠",
        layout="wide",
    )

    _init_state()
    store = get_store()

    _render_sidebar(store)

    upload_col, ask_col = st.columns([1, 1], gap="large")
    with upload_col:
        _render_upload_panel(store)
    with ask_col:
        _render_ask_panel(store)


main()
