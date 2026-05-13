def load_css():
    return """
    <style>

    /* Center content like modal */
    .block-container {
        max-width: 900px;
        margin: auto;
        padding-top: 40px;
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 10px;
        padding: 0.5em 1em;
        border: 1px solid #444;
        background-color: #111;
        color: white;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        border: 1px solid #888;
        background-color: #222;
    }

    /* Selected button (we'll inject per-key styles dynamically) */
    button.selected {
        background-color: #E50914 !important;
        border: 1px solid #E50914 !important;
        color: white !important;
    }

    </style>
    """