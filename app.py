```python
import streamlit as st
import weasyprint
import json
import io
import random
import html
import tempfile
import os

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from google import genai
from pdf2image import convert_from_bytes


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Math Script Examiner",
    page_icon="📝",
    layout="centered"
)

st.title("📝 AI Math Script Examiner")

st.write(
    "Upload handwritten student scripts (JPG, PNG, or PDF) "
    "to generate a marked mathematics report."
)


# ============================================================
# GEMINI API KEY
# ============================================================

raw_keys = (
    st.secrets.get("GEMINI_API_KEYS")
    or st.secrets.get("GEMINI_API_KEY")
)

if not raw_keys:
    st.error("No API key found in Streamlit Secrets!")
    st.stop()

if isinstance(raw_keys, list):
    api_keys_list = [
        str(k).strip()
        for k in raw_keys
        if str(k).strip()
    ]

elif isinstance(raw_keys, str):
    api_keys_list = [
        k.strip()
        for k in raw_keys.split(",")
        if k.strip()
    ]

else:
    api_keys_list = [str(raw_keys).strip()]


# ============================================================
# SESSION STATE
# Prevent unnecessary Gemini calls after result is generated
# ============================================================

if "student_data" not in st.session_state:
    st.session_state.student_data = None

if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_files = st.file_uploader(
    "Choose student script files (Images or PDFs)...",
    type=["jpg", "jpeg", "png", "pdf"],
    accept_multiple_files=True
)


# ============================================================
# HELPER: SAVE MATPLOTLIB FIGURE AS PNG
# ============================================================

def figure_to_base64(fig):

    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    import base64

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode()

    return (
        f"data:image/png;base64,{encoded}"
    )


# ============================================================
# HELPER: DRAW FUNCTION GRAPH
# ============================================================

def draw_function_graph(data):

    equation = data.get("equation", "")
    x_min = data.get("x_min", -10)
    x_max = data.get("x_max", 10)
    title = data.get("title", "Correct Graph")

    if not equation:
        return ""

    try:

        # Example:
        # y = x^2 + 2x - 3
        expression = equation.replace("^", "**")

        if "=" in expression:
            expression = expression.split("=")[1]

        x = np.linspace(
            float(x_min),
            float(x_max),
            400
        )

        safe_dict = {
            "x": x,
            "np": np,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "sqrt": np.sqrt,
            "abs": np.abs,
            "exp": np.exp,
            "log": np.log
        }

        y = eval(
            expression,
            {"__builtins__": {}},
            safe_dict
        )

        fig, ax = plt.subplots(
            figsize=(6, 4)
        )

        ax.axhline(0)
        ax.axvline(0)

        ax.plot(
            x,
            y
        )

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax.grid(True)

        return figure_to_base64(fig)

    except Exception:

        return ""


# ============================================================
# HELPER: DRAW POINT GRAPH
# ============================================================

def draw_points_graph(data):

    points = data.get("points", [])
    title = data.get("title", "Correct Graph")

    if not points:
        return ""

    try:

        x_values = [
            point["x"]
            for point in points
        ]

        y_values = [
            point["y"]
            for point in points
        ]

        fig, ax = plt.subplots(
            figsize=(6, 4)
        )

        ax.axhline(0)
        ax.axvline(0)

        ax.plot(
            x_values,
            y_values,
            marker="o"
        )

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax.grid(True)

        return figure_to_base64(fig)

    except Exception:

        return ""


# ============================================================
# HELPER: CREATE HTML TABLE
# ============================================================

def create_html_table(data):

    headers = data.get("headers", [])
    rows = data.get("rows", [])

    if not headers or not rows:
        return ""

    table_html = (
        '<table class="math-table">'
    )

    table_html += "<tr>"

    for header in headers:

        table_html += (
            f"<th>{html.escape(str(header))}</th>"
        )

    table_html += "</tr>"

    for row in rows:

        table_html += "<tr>"

        for cell in row:

            table_html += (
                f"<td>{html.escape(str(cell))}</td>"
            )

        table_html += "</tr>"

    table_html += "</table>"

    return table_html


# ============================================================
# HELPER: CREATE SIMPLE GEOMETRY DIAGRAM
# ============================================================

def draw_geometry(data):

    points = data.get("points", [])

    if len(points) < 2:
        return ""

    try:

        fig, ax = plt.subplots(
            figsize=(5, 4)
        )

        x_values = []
        y_values = []

        for point in points:

            x_values.append(point["x"])
            y_values.append(point["y"])

        # Close polygon
        if len(points) > 2:

            x_values.append(points[0]["x"])
            y_values.append(points[0]["y"])

        ax.plot(
            x_values,
            y_values,
            marker="o"
        )

        # Add labels
        for point in points:

            label = point.get(
                "label",
                ""
            )

            ax.text(
                point["x"],
                point["y"],
                f" {label}"
            )

        ax.set_aspect("equal")
        ax.grid(True)

        title = data.get(
            "title",
            "Correct Diagram"
        )

        ax.set_title(title)

        return figure_to_base64(fig)

    except Exception:

        return ""


# ============================================================
# GEMINI PROMPT
# ============================================================

prompt = """
You are an expert secondary-school mathematics examiner.

Analyze the uploaded student script carefully.

Your job is to:

1. Identify every question.
2. REPEAT the complete question or question instruction in the
   "question_text" field. This is important.
3. Read the student's working in the order written.
4. Mark the student's work step by step.
5. Identify incorrect working.
6. Give the correct mathematical working or answer.
7. Give a short reason for each error.
8. Award marks fairly according to mathematical method and answer.
9. If a question requires a table, graph, coordinate plot, or
   simple geometry diagram for correction, return STRUCTURED DATA
   describing what should be drawn.

IMPORTANT RULES:

- Return ONLY valid JSON.
- Do not use Markdown.
- Do not add commentary outside JSON.
- Only transcribe work that is clearly visible.
- Do not invent student work.
- Repeat the full question because the final PDF must show it.
- Keep explanations concise.
- Do not explain correct lines.
- For incorrect lines, explanations should normally be one sentence.

FOR CORRECT WORK:

Use:

{
    "text": "Student written line",
    "correct": true
}

FOR INCORRECT WORK:

Use:

{
    "text": "Student written line",
    "correct": false,
    "error_type": "Calculation Error",
    "correction": "Correct mathematical line",
    "explanation": "Brief reason for the error."
}

VISUAL CORRECTIONS:

If no visual correction is needed:

"visual": null

If a mathematical table is needed:

"visual": {
    "type": "table",
    "headers": ["Column 1", "Column 2"],
    "rows": [
        ["value", "value"]
    ]
}

If a graph of an equation is needed:

"visual": {
    "type": "function_graph",
    "title": "Correct Graph",
    "equation": "y = x^2 - 4",
    "x_min": -5,
    "x_max": 5
}

If a graph should be created from coordinate points:

"visual": {
    "type": "points_graph",
    "title": "Correct Graph",
    "points": [
        {"x": -2, "y": 4},
        {"x": 0, "y": 0},
        {"x": 2, "y": 4}
    ]
}

If a simple geometry diagram is needed:

"visual": {
    "type": "geometry",
    "title": "Correct Diagram",
    "points": [
        {"label": "A", "x": 0, "y": 0},
        {"label": "B", "x": 6, "y": 0},
        {"label": "C", "x": 3, "y": 4}
    ]
}

Return this JSON structure:

{
    "instruction": "Exam heading or instruction",
    "questions": [
        {
            "title": "Question 1",
            "question_text": "Complete question exactly or as clearly visible",
            "max_score": 4,
            "score": 2,
            "working": [
                {
                    "text": "Student written line",
                    "correct": true
                },
                {
                    "text": "Incorrect student line",
                    "correct": false,
                    "error_type": "Calculation Error",
                    "correction": "Correct mathematical line",
                    "explanation": "Brief reason."
                }
            ],
            "visual": null
        }
    ],
    "feedback": {
        "strengths": [
            "Strength 1"
        ],
        "improvements": [
            "Improvement 1"
        ]
    }
}
"""


# ============================================================
# PROCESS BUTTON
# ============================================================

if uploaded_files:

    if st.button(
        "Grade All Scripts & Generate Combined PDF",
        type="primary"
    ):

        with st.spinner(
            "Processing mathematics script..."
        ):

            try:

                selected_key = random.choice(
                    api_keys_list
                )

                client = genai.Client(
                    api_key=selected_key
                )

                all_images = []

                for file in uploaded_files:

                    file_bytes = file.read()

                    # PDF
                    if file.name.lower().endswith(
                        ".pdf"
                    ):

                        pdf_images = (
                            convert_from_bytes(
                                file_bytes
                            )
                        )

                        for img in pdf_images:

                            img = img.convert(
                                "RGB"
                            )

                            img.thumbnail(
                                (1280, 1280)
                            )

                            all_images.append(
                                img
                            )

                    # IMAGE
                    else:

                        img = Image.open(
                            io.BytesIO(
                                file_bytes
                            )
                        ).convert(
                            "RGB"
                        )

                        img.thumbnail(
                            (1280, 1280)
                        )

                        all_images.append(
                            img
                        )

                st.info(
                    f"Loaded {len(all_images)} page(s)."
                )

                contents_payload = (
                    all_images + [prompt]
                )

                response = (
                    client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=contents_payload,
                        config={
                            "response_mime_type":
                                "application/json",

                            "max_output_tokens":
                                8192
                        }
                    )
                )

                student_data = json.loads(
                    response.text
                )

                st.session_state.student_data = (
                    student_data
                )

                # =================================================
                # GENERATE PDF HTML
                # =================================================

                total_score = sum(
                    q.get("score", 0)
                    for q in student_data.get(
                        "questions",
                        []
                    )
                )

                max_score = sum(
                    q.get("max_score", 0)
                    for q in student_data.get(
                        "questions",
                        []
                    )
                )

                percentage = (
                    round(
                        total_score /
                        max_score * 100,
                        1
                    )
                    if max_score > 0
                    else 0
                )

                questions_html = ""

                for question in student_data.get(
                    "questions",
                    []
                ):

                    working_html = ""

                    for step_number, line in enumerate(
                        question.get(
                            "working",
                            []
                        ),
                        start=1
                    ):

                        text = html.escape(
                            str(
                                line.get(
                                    "text",
                                    ""
                                )
                            )
                        )

                        correct = line.get(
                            "correct",
                            False
                        )

                        if correct:

                            working_html += f"""
                            <div class="working-line">
                                {text}
                                <span class="tick">✓</span>
                            </div>
                            """

                        else:

                            error_type = html.escape(
                                str(
                                    line.get(
                                        "error_type",
                                        "Error"
                                    )
                                )
                            )

                            correction = html.escape(
                                str(
                                    line.get(
                                        "correction",
                                        ""
                                    )
                                )
                            )

                            explanation = html.escape(
                                str(
                                    line.get(
                                        "explanation",
                                        ""
                                    )
                                )
                            )

                            working_html += f"""
                            <div class="working-line wrong">
                                <span>{text}</span>
                                <span class="cross">✗</span>
                                <span class="error-label">
                                    {error_type}
                                </span>
                            </div>

                            <div class="correction">
                                Correct: {correction}
                            </div>

                            <div class="explanation">
                                <strong>
                                    Step {step_number}:
                                </strong>
                                {explanation}
                            </div>
                            """

                    # =============================================
                    # CREATE VISUAL
                    # =============================================

                    visual_html = ""

                    visual = question.get(
                        "visual"
                    )

                    if visual:

                        visual_type = visual.get(
                            "type"
                        )

                        if visual_type == "table":

                            visual_html = (
                                create_html_table(
                                    visual
                                )
                            )

                        elif visual_type == (
                            "function_graph"
                        ):

                            image_data = (
                                draw_function_graph(
                                    visual
                                )
                            )

                            if image_data:

                                visual_html = f"""
                                <img
                                    src="{image_data}"
                                    class="graph-image"
                                >
                                """

                        elif visual_type == (
                            "points_graph"
                        ):

                            image_data = (
                                draw_points_graph(
                                    visual
                                )
                            )

                            if image_data:

                                visual_html = f"""
                                <img
                                    src="{image_data}"
                                    class="graph-image"
                                >
                                """

                        elif visual_type == (
                            "geometry"
                        ):

                            image_data = (
                                draw_geometry(
                                    visual
                                )
                            )

                            if image_data:

                                visual_html = f"""
                                <img
                                    src="{image_data}"
                                    class="graph-image"
                                >
                                """

                    questions_html += f"""
                    <div class="question-block">

                        <div class="question-title">
                            {html.escape(
                                str(
                                    question.get(
                                        "title",
                                        "Question"
                                    )
                                )
                            )}
                        </div>

                        <div class="question-text">
                            <strong>Question:</strong><br>
                            {html.escape(
                                str(
                                    question.get(
                                        "question_text",
                                        ""
                                    )
                                )
                            )}
                        </div>

                        <div class="working-box">
                            <strong>
                                Student's Working:
                            </strong>

                            {working_html}
                        </div>

                        <div class="score">
                            Score:
                            {question.get("score", 0)}
                            /
                            {question.get("max_score", 0)}
                        </div>

                        {visual_html}

                    </div>
                    """

                # =================================================
                # FEEDBACK
                # =================================================

                feedback = student_data.get(
                    "feedback",
                    {}
                )

                strengths_html = "".join(
                    f"<li>{html.escape(str(x))}</li>"
                    for x in feedback.get(
                        "strengths",
                        []
                    )
                )

                improvements_html = "".join(
                    f"<li>{html.escape(str(x))}</li>"
                    for x in feedback.get(
                        "improvements",
                        []
                    )
                )

                # =================================================
                # FINAL HTML
                # =================================================

                html_content = f"""
                <!DOCTYPE html>

                <html>

                <head>

                <style>

                @page {{
                    size: A4;
                    margin: 15mm;
                }}

                body {{
                    font-family: Arial;
                    font-size: 11pt;
                    line-height: 1.5;
                }}

                .summary {{
                    border: 2px solid #d90000;
                    padding: 15px;
                    text-align: center;
                    margin-bottom: 20px;
                }}

                .question-block {{
                    border: 1px solid #cccccc;
                    padding: 15px;
                    margin-bottom: 20px;
                }}

                .question-title {{
                    font-weight: bold;
                    font-size: 13pt;
                    margin-bottom: 8px;
                }}

                .question-text {{
                    background: #f2f2f2;
                    padding: 10px;
                    margin-bottom: 12px;
                    white-space: pre-wrap;
                }}

                .working-box {{
                    background: #f8f9ff;
                    padding: 10px;
                }}

                .working-line {{
                    margin-top: 8px;
                }}

                .wrong {{
                    text-decoration: line-through;
                }}

                .tick {{
                    color: green;
                    font-weight: bold;
                    margin-left: 8px;
                }}

                .cross {{
                    color: red;
                    font-weight: bold;
                    margin-left: 8px;
                }}

                .error-label {{
                    color: #d90000;
                    font-size: 9pt;
                    margin-left: 5px;
                }}

                .correction {{
                    color: #d90000;
                    font-weight: bold;
                    margin: 5px 0;
                }}

                .explanation {{
                    background: #fff9e6;
                    padding: 8px;
                    margin-bottom: 8px;
                    font-size: 10pt;
                }}

                .score {{
                    margin-top: 12px;
                    font-size: 13pt;
                    font-weight: bold;
                    color: #d90000;
                }}

                .math-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 15px;
                }}

                .math-table th,
                .math-table td {{
                    border: 1px solid black;
                    padding: 6px;
                    text-align: center;
                }}

                .graph-image {{
                    max-width: 100%;
                    margin-top: 15px;
                }}

                .feedback {{
                    border: 2px solid #d90000;
                    padding: 15px;
                }}

                </style>

                </head>

                <body>

                <div class="summary">

                    <h2>
                        Mathematics Script Examination Result
                    </h2>

                    <strong>
                        TOTAL SCORE:
                        {total_score}/{max_score}
                    </strong>

                    <br>

                    <strong>
                        PERCENTAGE:
                        {percentage}%
                    </strong>

                </div>

                <div>
                    <strong>
                        {html.escape(
                            str(
                                student_data.get(
                                    "instruction",
                                    ""
                                )
                            )
                        )}
                    </strong>
                </div>

                {questions_html}

                <div class="feedback">

                    <h3>
                        Examiner's Feedback
                    </h3>

                    <strong>
                        Key Strengths
                    </strong>

                    <ul>
                        {strengths_html}
                    </ul>

                    <strong>
                        Areas for Improvement
                    </strong>

                    <ul>
                        {improvements_html}
                    </ul>

                </div>

                </body>

                </html>
                """

                pdf_bytes = (
                    weasyprint.HTML(
                        string=html_content
                    ).write_pdf()
                )

                st.session_state.pdf_bytes = (
                    pdf_bytes
                )

                st.success(
                    "Evaluation complete!"
                )

            except Exception as e:

                st.error(
                    f"Error processing scripts: {e}"
                )


# ============================================================
# DOWNLOAD WITHOUT CALLING GEMINI AGAIN
# ============================================================

if st.session_state.pdf_bytes:

    st.download_button(
        label="📄 Download Marked PDF",
        data=st.session_state.pdf_bytes,
        file_name="marked_math_script.pdf",
        mime="application/pdf"
    )
```

