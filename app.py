"""Single-Page Agentic RAG PDF Chat Application."""
import time
from pathlib import Path
import streamlit as st

from config import (
    UPLOADS_DIR,
    LLM_MODEL,
    EMBEDDING_MODEL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_TEMPERATURE,
)
from utils.helpers import inject_linear_theme
from utils.pdf_loader import load_pdf_document
from utils.chunker import chunk_documents
from utils.chroma_store import ChromaStoreManager
from utils.ollama_client import check_ollama_status
from graph import run_agentic_rag

st.set_page_config(
    page_title="Agentic PDF Chat",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_linear_theme()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

store_manager = ChromaStoreManager()


def main():
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
                <div style="background-color: #5e6ad2; width: 28px; height: 28px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-weight: bold; color: white;">A</div>
                <div>
                    <div style="font-weight: 600; font-size: 16px; color: #f7f8f8; line-height: 1.1;">Agentic PDF Chat</div>
                    <div style="font-size: 11px; color: #8a8f98;">Linear Local RAG System</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<hr style='border: 0; border-top: 1px solid #23252a; margin: 16px 0;'>", unsafe_allow_html=True)

        ollama_online = check_ollama_status()
        if ollama_online:
            st.markdown(
                '<div class="linear-pill linear-pill-success" style="margin-bottom: 16px;">● Ollama Connected</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="linear-pill" style="margin-bottom: 16px; color: #e54d42; border-color: rgba(229,77,66,0.3); background-color: rgba(229,77,66,0.15);">● Ollama Offline</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<h4 style='font-size: 13px; color: #8a8f98; text-transform: uppercase; letter-spacing: 0.05em;'>Upload Documents</h4>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            st.markdown("<div style='font-size: 12px; color: #d0d6e0; margin-top: 6px; margin-bottom: 8px;'>Uploaded PDFs:</div>", unsafe_allow_html=True)
            for file in uploaded_files:
                st.markdown(f"<div style='font-size: 12px; color: #8a8f98; padding: 2px 0;'>📄 {file.name}</div>", unsafe_allow_html=True)

        col_idx, col_clr = st.columns(2)
        with col_idx:
            btn_index = st.button("Index Documents", type="primary", use_container_width=True)
        with col_clr:
            btn_clear = st.button("Clear DB", use_container_width=True)

        if btn_index:
            if not uploaded_files:
                st.toast("⚠️ Please select at least one PDF to index.", icon="⚠️")
            else:
                with st.spinner("Processing & indexing PDFs..."):
                    all_nodes = []
                    indexed_names = []
                    for file in uploaded_files:
                        file_path = UPLOADS_DIR / file.name
                        with open(file_path, "wb") as f:
                            f.write(file.getvalue())

                        docs = load_pdf_document(file_path_or_bytes=file.getvalue(), filename=file.name)
                        nodes = chunk_documents(
                            docs,
                            chunk_size=st.session_state.get("cfg_chunk_size", DEFAULT_CHUNK_SIZE),
                            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                        )
                        all_nodes.extend(nodes)
                        indexed_names.append(file.name)

                    added_count = store_manager.index_nodes(all_nodes)
                    st.session_state.indexed_files = list(set(st.session_state.indexed_files + indexed_names))
                    st.toast(f"✅ Indexed {added_count} chunks!", icon="✅")
                    st.rerun()

        if btn_clear:
            store_manager.clear_database()
            st.session_state.indexed_files = []
            st.session_state.chat_history = []
            st.toast("🗑️ Database cleared.", icon="🗑️")
            st.rerun()

        st.markdown("<hr style='border: 0; border-top: 1px solid #23252a; margin: 18px 0;'>", unsafe_allow_html=True)

        st.markdown("<h4 style='font-size: 13px; color: #8a8f98; text-transform: uppercase; letter-spacing: 0.05em;'>Database Statistics</h4>", unsafe_allow_html=True)
        stats = store_manager.get_statistics()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""
                <div class="linear-stat-card">
                    <div class="linear-stat-value">{stats['documents']}</div>
                    <div class="linear-stat-label">Documents</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="linear-stat-card" style="margin-top: 8px;">
                    <div class="linear-stat-value">{stats['chunks']}</div>
                    <div class="linear-stat-label">Chunks</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f"""
                <div class="linear-stat-card">
                    <div class="linear-stat-value">{stats['pages']}</div>
                    <div class="linear-stat-label">Pages</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="linear-stat-card" style="margin-top: 8px;">
                    <div class="linear-stat-value">{stats['embeddings']}</div>
                    <div class="linear-stat-label">Embeddings</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<hr style='border: 0; border-top: 1px solid #23252a; margin: 18px 0;'>", unsafe_allow_html=True)

        st.markdown("<h4 style='font-size: 13px; color: #8a8f98; text-transform: uppercase; letter-spacing: 0.05em;'>Model Information</h4>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="font-size: 12px; color: #d0d6e0; background: #16171d; border: 1px solid #23252a; padding: 10px 12px; border-radius: 8px; margin-bottom: 12px;">
                <div style="color: #8a8f98; font-size: 11px;">Chat Model</div>
                <div style="font-weight: 600; color: #f7f8f8; font-family: 'JetBrains Mono';">{LLM_MODEL}</div>
                <div style="color: #8a8f98; font-size: 11px; margin-top: 6px;">Embedding Model</div>
                <div style="font-weight: 600; color: #f7f8f8; font-family: 'JetBrains Mono';">{EMBEDDING_MODEL}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<hr style='border: 0; border-top: 1px solid #23252a; margin: 18px 0;'>", unsafe_allow_html=True)

        st.markdown("<h4 style='font-size: 13px; color: #8a8f98; text-transform: uppercase; letter-spacing: 0.05em;'>Settings</h4>", unsafe_allow_html=True)
        cfg_chunk_size = st.slider("Chunk Size", min_value=256, max_value=1024, value=DEFAULT_CHUNK_SIZE, step=64, key="cfg_chunk_size")
        cfg_top_k = st.slider("Top-K", min_value=1, max_value=10, value=DEFAULT_TOP_K, step=1, key="cfg_top_k")
        cfg_temp = st.slider("Temperature", min_value=0.0, max_value=1.0, value=DEFAULT_TEMPERATURE, step=0.05, key="cfg_temp")

    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 600; letter-spacing: -0.03em;">Agentic PDF Chat</h1>
            <p style="margin: 4px 0 0 0; color: #8a8f98; font-size: 1.05rem;">Chat with your PDFs completely offline.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if "confidence" in msg and msg.get("role") == "assistant":
                conf = msg["confidence"]
                conf_color = "#27a644" if conf >= 75 else "#5e6ad2"
                st.markdown(
                    f"""
                    <div style="margin-top: 10px; display: inline-flex; align-items: center; gap: 8px; font-size: 12px; background-color: #16171d; border: 1px solid #23252a; padding: 4px 10px; border-radius: 9999px;">
                        <span style="color: #8a8f98;">Confidence:</span>
                        <strong style="color: {conf_color}; font-family: 'JetBrains Mono';">{conf:.0f}%</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if "sources" in msg and msg["sources"]:
                with st.expander("📚 Sources", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(
                            f"""
                            <div style="background-color: #0e0f12; border: 1px solid #23252a; border-radius: 8px; padding: 10px 12px; margin-bottom: 8px;">
                                <div style="font-weight: 600; font-size: 13px; color: #f7f8f8;">{src['filename']} — <span style="color: #8a8f98;">Page {src['page_number']}</span></div>
                                <div style="font-size: 12px; color: #d0d6e0; font-family: 'JetBrains Mono'; margin-top: 4px; line-height: 1.4;">"{src['snippet']}"</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            if "steps" in msg and msg["steps"]:
                with st.expander("⚙️ Workflow Status", expanded=False):
                    for step in msg["steps"]:
                        st.markdown(f"<div style='font-size: 12px; color: #d0d6e0; padding: 2px 0;'>✓ <strong>{step['name']}:</strong> {step['detail']}</div>", unsafe_allow_html=True)

    user_query = st.chat_input("Ask a question about your uploaded PDFs...")

    if user_query:
        if stats["chunks"] == 0:
            st.toast("⚠️ No documents indexed in ChromaDB yet.", icon="⚠️")

        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.status("Executing Workflow...", expanded=True) as status_box:
                st.write("🔄 Initializing state graph...")
                time.sleep(0.1)

                result = run_agentic_rag(
                    question=user_query,
                    top_k=cfg_top_k,
                    temperature=cfg_temp,
                )

                steps = result.get("processing_steps", [])
                for s in steps:
                    st.write(f"✓ **{s['name']}:** {s['detail']}")

                status_box.update(label="Workflow Complete", state="complete", expanded=False)

            answer_text = result.get("answer", "No answer generated.")
            confidence = result.get("confidence", 85.0)
            sources = result.get("sources", [])
            workflow_steps = result.get("processing_steps", [])

            response_placeholder = st.empty()
            full_response = ""
            for chunk in answer_text.split():
                full_response += chunk + " "
                response_placeholder.markdown(full_response + "▌")
                time.sleep(0.01)
            response_placeholder.markdown(full_response)

            conf_color = "#27a644" if confidence >= 75 else "#5e6ad2"
            st.markdown(
                f"""
                <div style="margin-top: 12px; display: inline-flex; align-items: center; gap: 8px; font-size: 12px; background-color: #16171d; border: 1px solid #23252a; padding: 4px 10px; border-radius: 9999px;">
                    <span style="color: #8a8f98;">Confidence:</span>
                    <strong style="color: {conf_color}; font-family: 'JetBrains Mono';">{confidence:.0f}%</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if sources:
                with st.expander("📚 Sources", expanded=True):
                    for src in sources:
                        st.markdown(
                            f"""
                            <div style="background-color: #0e0f12; border: 1px solid #23252a; border-radius: 8px; padding: 10px 12px; margin-bottom: 8px;">
                                <div style="font-weight: 600; font-size: 13px; color: #f7f8f8;">{src['filename']} — <span style="color: #8a8f98;">Page {src['page_number']}</span></div>
                                <div style="font-size: 12px; color: #d0d6e0; font-family: 'JetBrains Mono'; margin-top: 4px; line-height: 1.4;">"{src['snippet']}"</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": full_response,
                "confidence": confidence,
                "sources": sources,
                "steps": workflow_steps,
            })


if __name__ == "__main__":
    main()
