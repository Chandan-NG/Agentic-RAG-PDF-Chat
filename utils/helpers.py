"""Helper functions for loading prompt files and injecting Linear Design System CSS."""
from pathlib import Path
import streamlit as st
from config import (
    PROMPTS_DIR,
    COLOR_CANVAS,
    COLOR_SURFACE_1,
    COLOR_SURFACE_2,
    COLOR_SURFACE_3,
    COLOR_SURFACE_4,
    COLOR_HAIRLINE,
    COLOR_HAIRLINE_STRONG,
    COLOR_HAIRLINE_TERTIARY,
    COLOR_INK,
    COLOR_INK_MUTED,
    COLOR_INK_SUBTLE,
    COLOR_INK_TERTIARY,
    COLOR_PRIMARY,
    COLOR_PRIMARY_HOVER,
    COLOR_PRIMARY_FOCUS,
    COLOR_SEMANTIC_SUCCESS,
)


def load_prompt(prompt_name: str) -> str:
    """Load a markdown prompt template from the prompts directory."""
    path = PROMPTS_DIR / prompt_name
    if not path.exists():
        raise FileNotFoundError(f"Prompt template '{prompt_name}' not found at {path}")
    return path.read_text(encoding="utf-8")


def inject_linear_theme():
    """Inject Linear Design System styling into the Streamlit app.

    Enforces strictly the tokens and principles defined in DESIGN.md:
    - Canvas: #010102 (near black with faint blue tint)
    - Surface Ladder: #0e0f12 -> #16171d -> #1f2129 -> #282a36
    - Accent: Lavender #5e6ad2 (#828fff hover)
    - Hairline Borders: #23252a
    - Typography & Radius: Inter / SF Pro Display, 12px rounded cards, 8px inputs/buttons.
    - No gradients, no glassmorphism, no bright colors.
    """
    css = f"""
    <style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Root & Page Background */
    html, body, [data-testid="stAppViewContainer"], .stAppViewContainer {{
        background-color: {COLOR_CANVAS} !important;
        color: {COLOR_INK} !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display', system-ui, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }}

    [data-testid="stHeader"] {{
        background-color: {COLOR_CANVAS} !important;
        border-bottom: 1px solid {COLOR_HAIRLINE} !important;
    }}

    /* Main Container Padding */
    .main .block-container {{
        max-width: 1080px;
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: {COLOR_SURFACE_1} !important;
        border-right: 1px solid {COLOR_HAIRLINE} !important;
    }}

    [data-testid="stSidebar"] .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
    }}

    /* Headers & Typography */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Inter', sans-serif !important;
        color: {COLOR_INK} !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }}

    h1 {{
        font-size: 2.25rem !important;
        line-height: 1.15 !important;
        letter-spacing: -0.03em !important;
    }}

    p, span, label, div {{
        color: {COLOR_INK} !important;
    }}

    .stCaption, .stSubheaderCaption {{
        color: {COLOR_INK_SUBTLE} !important;
    }}

    /* Linear Surface Cards */
    .linear-card {{
        background-color: {COLOR_SURFACE_1};
        border: 1px solid {COLOR_HAIRLINE};
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 16px;
        transition: border-color 0.15s ease, background-color 0.15s ease;
    }}

    .linear-card:hover {{
        border-color: {COLOR_HAIRLINE_STRONG};
        background-color: {COLOR_SURFACE_2};
    }}

    .linear-stat-card {{
        background-color: {COLOR_SURFACE_2};
        border: 1px solid {COLOR_HAIRLINE};
        border-radius: 8px;
        padding: 12px 14px;
        text-align: center;
    }}

    .linear-stat-value {{
        font-size: 1.35rem;
        font-weight: 600;
        color: {COLOR_INK};
        font-family: 'JetBrains Mono', monospace;
    }}

    .linear-stat-label {{
        font-size: 0.75rem;
        color: {COLOR_INK_SUBTLE};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 2px;
    }}

    /* Secondary Buttons */
    .stButton > button {{
        background-color: {COLOR_SURFACE_2} !important;
        color: {COLOR_INK} !important;
        border: 1px solid {COLOR_HAIRLINE} !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        padding: 8px 16px !important;
        transition: all 0.15s ease !important;
    }}

    .stButton > button:hover {{
        background-color: {COLOR_SURFACE_3} !important;
        border-color: {COLOR_HAIRLINE_STRONG} !important;
        color: {COLOR_INK} !important;
    }}

    .stButton > button:focus {{
        border-color: {COLOR_PRIMARY_FOCUS} !important;
        box-shadow: 0 0 0 2px rgba(94, 106, 210, 0.3) !important;
    }}

    /* Primary Accent Button (Lavender #5e6ad2) */
    div[data-testid="stButton"] button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {{
        background-color: {COLOR_PRIMARY} !important;
        color: #ffffff !important;
        border: 1px solid {COLOR_PRIMARY} !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
    }}

    div[data-testid="stButton"] button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {{
        background-color: {COLOR_PRIMARY_HOVER} !important;
        border-color: {COLOR_PRIMARY_HOVER} !important;
        color: #ffffff !important;
    }}

    /* Inputs & Text Areas */
    .stTextInput input, .stSelectbox select, .stNumberInput input, .stTextArea textarea {{
        background-color: {COLOR_SURFACE_1} !important;
        border: 1px solid {COLOR_HAIRLINE} !important;
        border-radius: 8px !important;
        color: {COLOR_INK} !important;
        font-size: 14px !important;
        padding: 8px 12px !important;
    }}

    .stTextInput input:focus, .stSelectbox select:focus, .stTextArea textarea:focus {{
        border-color: {COLOR_PRIMARY_FOCUS} !important;
        box-shadow: 0 0 0 2px rgba(94, 106, 210, 0.25) !important;
    }}

    /* Sliders - Override default red with Linear Lavender */
    div[data-baseweb="slider"] [role="slider"] {{
        background-color: {COLOR_PRIMARY} !important;
        border-color: {COLOR_PRIMARY} !important;
    }}

    div[data-baseweb="slider"] div[style*="background-color: rgb"] {{
        background-color: {COLOR_PRIMARY} !important;
    }}

    .stSlider label {{
        color: {COLOR_INK_MUTED} !important;
        font-size: 13px !important;
    }}

    /* Chat Input Bar */
    [data-testid="stChatInput"] {{
        background-color: {COLOR_SURFACE_1} !important;
        border: 1px solid {COLOR_HAIRLINE_STRONG} !important;
        border-radius: 10px !important;
    }}

    [data-testid="stChatInput"] textarea {{
        background-color: transparent !important;
        color: {COLOR_INK} !important;
        border: none !important;
    }}

    /* Chat Message Bubbles */
    [data-testid="stChatMessage"] {{
        background-color: {COLOR_SURFACE_1} !important;
        border: 1px solid {COLOR_HAIRLINE} !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        margin-bottom: 12px !important;
    }}

    /* Expanders & Accordions */
    .stExpander {{
        background-color: {COLOR_SURFACE_1} !important;
        border: 1px solid {COLOR_HAIRLINE} !important;
        border-radius: 10px !important;
        margin-bottom: 10px !important;
    }}

    .stExpander summary {{
        color: {COLOR_INK_MUTED} !important;
        font-weight: 500 !important;
    }}

    /* Badges & Status Pills */
    .linear-pill {{
        display: inline-block;
        background-color: {COLOR_SURFACE_2};
        border: 1px solid {COLOR_HAIRLINE};
        border-radius: 9999px;
        padding: 3px 10px;
        font-size: 12px;
        font-weight: 500;
        color: {COLOR_INK_MUTED};
    }}

    .linear-pill-success {{
        background-color: rgba(39, 166, 68, 0.15);
        border-color: rgba(39, 166, 68, 0.3);
        color: {COLOR_SEMANTIC_SUCCESS};
    }}

    .linear-pill-primary {{
        background-color: rgba(94, 106, 210, 0.15);
        border-color: rgba(94, 106, 210, 0.3);
        color: {COLOR_PRIMARY_HOVER};
    }}

    /* Status Containers */
    [data-testid="stStatusWidget"] {{
        background-color: {COLOR_SURFACE_1} !important;
        border: 1px solid {COLOR_HAIRLINE} !important;
        border-radius: 8px !important;
    }}

    /* Code Snippets */
    code, pre {{
        font-family: 'JetBrains Mono', monospace !important;
        background-color: {COLOR_SURFACE_2} !important;
        color: {COLOR_INK_MUTED} !important;
        border: 1px solid {COLOR_HAIRLINE} !important;
        border-radius: 6px !important;
    }}

    /* File Uploader */
    [data-testid="stFileUploader"] {{
        background-color: {COLOR_SURFACE_1} !important;
        border: 1px dashed {COLOR_HAIRLINE_STRONG} !important;
        border-radius: 10px !important;
        padding: 12px !important;
    }}

    /* Toast Notifications */
    [data-testid="stToast"] {{
        background-color: {COLOR_SURFACE_2} !important;
        border: 1px solid {COLOR_HAIRLINE_STRONG} !important;
        color: {COLOR_INK} !important;
        border-radius: 8px !important;
    }}

    /* Hide Streamlit Default Branding Footer */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
