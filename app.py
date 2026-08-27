import streamlit as st
import weasyprint
import json
import io
import random
import html
import base64

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
    "to generate a single combined annotated PDF report."
)


# ============================================================
# FETCH API KEY(S)
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
# ============================================================

if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_files = st.file_uploader(
    "Choose student script files (Images or PDFs)...",
    type=["jpg", "jpeg", "png", "pdf"],
    accept_multiple_files=True
)


# ============================================================
# HELPER: FIGURE TO BASE64
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

    buffer.seek(0)

    encoded = base64.b64encode(
        buffer.read()
    ).decode("utf-8")

    return "data:image/png;base64," + encoded


# ============================================================
# HELPER: FUNCTION GRAPH
# ============================================================

def draw_function_graph(data):

    try:

        equation = str(
            data.get("equation", "")
        ).strip()

        if not equation:
            return ""

        expression = equation.replace("^", "**")

        if "=" in expression:
            expression = expression.split(
                "=",
                1
            )[1].strip()

        x_min = float(
            data.get("x_min", -10)
        )

        x_max = float(
            data.get("x_max", 10)
        )

        x = np.linspace(
            x_min,
            x_max,
            500
        )

        safe_values = {
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
            safe_values
        )

        fig, ax = plt.subplots(
            figsize=(6, 4)
        )

        ax.axhline(0)

        ax.axvline(0)

        ax.plot(x, y)

        ax.grid(True)

        ax.set_xlabel("x")

        ax.set_ylabel("y")

        ax.set_title(
            data.get(
                "title",
                "Correct Graph"
            )
        )

        return figure_to_base64(fig)

    except Exception:
        return ""


# ============================================================
# HELPER: POINT GRAPH
# ============================================================

def draw_points_graph(data):

    try:

        points = data.get(
            "points",
            []
        )

        if not points:
            return ""

        x_values = [
            float(p["x"])
            for p in points
        ]

        y_values = [
            float(p["y"])
            for p in points
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

        ax.grid(True)

        ax.set_xlabel("x")

        ax.set_ylabel("y")

        ax.set_title(
            data.get(
                "title",
                "Correct Graph"
            )
        )

        return figure_to_base64(fig)

    except Exception:
        return ""


# ============================================================
# HELPER: HTML TABLE
# ============================================================

def create_html_table(data):

    headers = data.get(
        "headers",
        []
    )

    rows = data.get(
        "rows",
        []
    )

    if not headers:
        return ""

    table_html = '<table class="math-table">'

    table_html += "<tr>"

    for header in headers:

        table_html += (
            "<th>"
            + html.escape(str(header))
            + "</th>"
        )

    table_html += "</tr>"

    for row in rows:

        table_html += "<tr>"

        for cell in row:

            table_html += (
                "<td>"
                + html.escape(str(cell))
                + "</td>"
            )

        table_html += "</tr>"

    table_html += "</table>"

    return table_html


# ============================================================
# PROMPT
# ============================================================

prompt = """
You are an expert secondary-school mathematics teacher and examiner.

Analyze the uploaded handwritten student mathematics script carefully.

Your task is to identify every question, repeat the question,
transcribe the student's work, mark it fairly and provide correct
working where necessary.

IMPORTANT EXAMINER RULES:

1. Repeat the COMPLETE question exactly as it appears in the script
   using the field "question_text".

2. Repeating the question is essential.

3. Do not invent student working that is not visible.

4. Transcribe the student's mathematical working in the order written.

5. Award marks fairly for:
   - correct method
   - correct mathematical steps
   - correct calculations
   - correct final answer

6. If a question is fully correct:
   set "status" to "correct"
   and return an empty "correct_solution" list.

7. If a question contains errors:
   set "status" to "incorrect"
   and provide the complete correct working in "correct_solution".

8. If a question was not attempted:
   set "status" to "undone"
   and provide the complete correct working.

9. Every incorrect student line must contain:
   - error_type
   - explanation

10. Do not provide a correction for a fully correct question.

11. The correct_solution must be a clear step-by-step mathematical
    solution.

12. Keep explanations concise.

VISUAL OUTPUT:

If the correct solution requires a table, graph or visual,
return the visual object.

If no visual is required:

"visual": null

TABLE FORMAT:

"visual": {
    "type": "table",
    "headers": ["Column 1", "Column 2"],
    "rows": [
        ["Value 1", "Value 2"]
    ]
}

FUNCTION GRAPH FORMAT:

"visual": {
    "type": "function_graph",
    "title": "Correct Graph",
    "equation": "y = x^2 - 4",
    "x_min": -5,
    "x_max": 5
}

POINT GRAPH FORMAT:

"visual": {
    "type": "points_graph",
    "title": "Correct Graph",
    "points": [
        {"x": -2, "y": 4},
        {"x": 0, "y": 0},
        {"x": 2, "y": 4}
    ]
}

Return ONLY one valid JSON object.

Use exactly this structure:

{
    "instruction": "Exam heading or instructions",

    "questions": [
        {
            "title": "Question 1",

            "question_text": "Complete question exactly as read",

            "max_score": 4,

            "score": 2,

            "status": "incorrect",

            "working": [
                {
                    "text": "Student written mathematical line",
                    "correct": true
                },
                {
                    "text": "Incorrect student line",
                    "correct": false,
                    "error_type": "Calculation Error",
                    "explanation": "Brief reason why the step is incorrect."
                }
            ],

            "correct_solution": [
                "Correct step 1",
                "Correct step 2",
                "Correct final answer"
            ],

            "visual": null
        }
    ],

    "feedback": {
        "strengths": [
            "Key strength"
        ],
        "improvements": [
            "Key improvement"
        ]
    }
}
"""


# ============================================================
# PROCESS FILES
# ============================================================

if uploaded_files:

    if st.button(
        "Grade All Scripts & Generate Combined PDF",
        type="primary"
    ):

        with st.spinner(
            "Processing files, evaluating handwriting, and compiling PDF..."
        ):

            try:

                selected_key = random.choice(
                    api_keys_list
                )

                client = genai.Client(
                    api_key=selected_key
                )

                all_images = []

                # --------------------------------------------
                # CONVERT FILES TO IMAGES
                # --------------------------------------------

                for file in uploaded_files:

                    file_bytes = file.read()

                    if file.name.lower().endswith(".pdf"):

                        pdf_images = convert_from_bytes(
                            file_bytes
                        )

                        for img in pdf_images:

                            img = img.convert("RGB")

                            img.thumbnail(
                                (1280, 1280)
                            )

                            all_images.append(img)

                    else:

                        img = Image.open(
                            io.BytesIO(file_bytes)
                        ).convert("RGB")

                        img.thumbnail(
                            (1280, 1280)
                        )

                        all_images.append(img)

                st.info(
                    f"Loaded {len(all_images)} total page(s). Evaluating..."
                )

                # --------------------------------------------
                # SEND TO GEMINI
                # --------------------------------------------

                contents_payload = (
                    all_images + [prompt]
                )

                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=contents_payload,
                    config={
                        "response_mime_type": "application/json",
                        "max_output_tokens": 8192
                    }
                )

                student_data = json.loads(
                    response.text.strip()
                )

                # --------------------------------------------
                # TOTAL SCORE
                # --------------------------------------------

                total_score = sum(
                    item.get("score", 0)
                    for item in student_data.get(
                        "questions",
                        []
                    )
                )

                max_score = sum(
                    item.get("max_score", 0)
                    for item in student_data.get(
                        "questions",
                        []
                    )
                )

                percentage = (
                    round(
                        (total_score / max_score) * 100,
                        1
                    )
                    if max_score > 0
                    else 0
                )

                # --------------------------------------------
                # QUESTIONS HTML
                # --------------------------------------------

                questions_html = ""

                for q in student_data.get(
                    "questions",
                    []
                ):

                    working_lines_html = ""

                    working_list = q.get(
                        "working",
                        []
                    )

                    # ========================================
                    # STUDENT WORK
                    # ========================================

                    if not working_list:

                        working_lines_html = """
                        <div class="student-line undone">
                            Not attempted
                        </div>
                        """

                    else:

                        for line in working_list:

                            text = html.escape(
                                str(
                                    line.get(
                                        "text",
                                        ""
                                    )
                                )
                            )

                            is_correct = line.get(
                                "correct",
                                False
                            )

                            if is_correct:

                                marking_html = (
                                    '<span class="tick">'
                                    '&#10003;'
                                    '</span>'
                                )

                            else:

                                marking_html = (
                                    '<span class="cross">'
                                    '&#10007;'
                                    '</span>'
                                )

                            error_html = ""

                            if (
                                not is_correct
                                and line.get(
                                    "error_type"
                                )
                            ):

                                error_html = f"""
                                <div class="marking-line">
                                    {html.escape(
                                        str(
                                            line.get(
                                                "error_type",
                                                ""
                                            )
                                        )
                                    )}
                                </div>
                                """

                            explanation_html = ""

                            if (
                                not is_correct
                                and line.get(
                                    "explanation"
                                )
                            ):

                                explanation_html = f"""
                                <div class="marking-explanation">
                                    {html.escape(
                                        str(
                                            line.get(
                                                "explanation",
                                                ""
                                            )
                                        )
                                    )}
                                </div>
                                """

                            working_lines_html += f"""

                            <div class="student-line">
                                <span class="student-text">
                                    {text}
                                </span>

                                <span class="mark-symbol">
                                    {marking_html}
                                </span>
                            </div>

                            {error_html}

                            {explanation_html}

                            """

                    # ========================================
                    # CORRECT WORKING
                    # ========================================

                    correct_working_html = ""

                    status = q.get(
                        "status",
                        "incorrect"
                    )

                    correct_solution = q.get(
                        "correct_solution",
                        []
                    )

                    if (
                        status in [
                            "incorrect",
                            "undone"
                        ]
                        and correct_solution
                    ):

                        correct_lines = ""

                        for step in correct_solution:

                            correct_lines += f"""
                            <div class="correct-line">
                                {html.escape(str(step))}
                            </div>
                            """

                        correct_working_html = f"""

                        <div class="correct-section">

                            <div class="correct-heading">
                                CORRECT WORKING
                            </div>

                            {correct_lines}

                        </div>

                        """

                    # ========================================
                    # VISUAL
                    # ========================================

                    visual_html = ""

                    visual = q.get(
                        "visual"
                    )

                    if visual:

                        visual_type = visual.get(
                            "type"
                        )

                        if visual_type == "table":

                            table_html = create_html_table(
                                visual
                            )

                            visual_html = f"""
                            <div class="visual-section">
                                {table_html}
                            </div>
                            """

                        elif visual_type == "function_graph":

                            graph = draw_function_graph(
                                visual
                            )

                            if graph:

                                visual_html = f"""
                                <div class="visual-section">
                                    <img
                                        src="{graph}"
                                        class="graph-image"
                                    >
                                </div>
                                """

                        elif visual_type == "points_graph":

                            graph = draw_points_graph(
                                visual
                            )

                            if graph:

                                visual_html = f"""
                                <div class="visual-section">
                                    <img
                                        src="{graph}"
                                        class="graph-image"
                                    >
                                </div>
                                """

                    # ========================================
                    # QUESTION CARD
                    # ========================================

                    questions_html += f"""

                    <div class="question-block">

                        <div class="question-title">

                            {html.escape(
                                str(
                                    q.get(
                                        "title",
                                        "Question"
                                    )
                                )
                            )}

                            &nbsp;&nbsp;&nbsp;

                            [
                            {q.get("max_score", 0)}
                            Marks
                            ]

                        </div>

                        <div class="question-text">

                            {html.escape(
                                str(
                                    q.get(
                                        "question_text",
                                        ""
                                    )
                                )
                            )}

                        </div>

                        <table class="script-table">

                            <tr>

                                <td class="script-cell work-cell">

                                    <div class="student-heading">
                                        STUDENT'S WORK
                                    </div>

                                    {working_lines_html}

                                    {correct_working_html}

                                    {visual_html}

                                </td>

                                <td class="script-cell marks-cell">

                                    <div class="marking-heading">
                                        MARKING
                                    </div>

                                    <div class="sub-score">

                                        {q.get("score", 0)}

                                        /

                                        {q.get("max_score", 0)}

                                    </div>

                                </td>

                            </tr>

                        </table>

                    </div>

                    """

                # --------------------------------------------
                # FEEDBACK
                # --------------------------------------------

                feedback = student_data.get(
                    "feedback",
                    {}
                )

                strengths_html = "".join(
                    [
                        "<li>"
                        + html.escape(str(item))
                        + "</li>"

                        for item in feedback.get(
                            "strengths",
                            []
                        )
                    ]
                )

                improvements_html = "".join(
                    [
                        "<li>"
                        + html.escape(str(item))
                        + "</li>"

                        for item in feedback.get(
                            "improvements",
                            []
                        )
                    ]
                )

                # ============================================
                # HTML AND CSS
                # ============================================

                html_content = f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<style>

@page {{
    size: A4;
    margin: 15mm;
    background-color: #fcfcfc;
}}

body {{
    font-family: Arial, sans-serif;
    color: #111;
    font-size: 11pt;
    line-height: 1.5;
}}


/* ========================================= */
/* SUMMARY HEADER */
/* ========================================= */

.summary-header {{
    border: 2px solid #d90000;
    background-color: #fff0f0;
    border-radius: 6px;
    padding: 12px 20px;
    margin-bottom: 20px;
    text-align: center;
}}

.summary-header h1 {{
    color: #d90000;
    margin: 0 0 8px 0;
    font-size: 16pt;
    text-transform: uppercase;
}}

.summary-stats {{
    font-size: 14pt;
    font-weight: bold;
    color: #b30000;
}}

.summary-stats span {{
    margin: 0 15px;
}}


/* ========================================= */
/* QUESTION CARD */
/* ========================================= */

.question-block {{
    background: #fff;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    padding: 15px;
    margin-bottom: 20px;
}}

.question-title {{
    color: #111;
    font-size: 11pt;
    font-weight: bold;
    margin-bottom: 6px;
    border-bottom: 1px dashed #ccc;
    padding-bottom: 5px;
}}

.question-text {{
    color: #111;
    font-size: 10.5pt;
    margin-bottom: 10px;
    white-space: pre-wrap;
}}


/* ========================================= */
/* TABLE LAYOUT */
/* ========================================= */

.script-table {{
    width: 100%;
    border-collapse: collapse;
}}

.script-cell {{
    vertical-align: top;
    padding: 4px;
}}


/* ========================================= */
/* STUDENT WORK - BLUE */
/* ========================================= */

.work-cell {{
    width: 75%;
    font-family: "Courier New", monospace;
    font-size: 10.5pt;
    color: #002b80;
    background-color: #f8f9ff;
    border-left: 3px solid #002b80;
    padding: 10px;
}}

.student-heading {{
    color: #002b80;
    font-weight: bold;
    font-family: Arial, sans-serif;
    margin-bottom: 6px;
}}

.student-line {{
    color: #002b80;
    margin-top: 6px;
    white-space: pre-wrap;
}}

.student-text {{
    color: #002b80;
}}

.undone {{
    font-style: italic;
}}


/* ========================================= */
/* MARKING - RED */
/* ========================================= */

.marks-cell {{
    width: 25%;
    text-align: right;
    padding-left: 10px;
}}

.marking-heading {{
    color: #d90000;
    font-weight: bold;
    font-family: Arial, sans-serif;
    font-size: 9.5pt;
    margin-bottom: 6px;
}}

.tick {{
    color: #d90000;
    font-weight: bold;
    font-size: 13pt;
}}

.cross {{
    color: #d90000;
    font-weight: bold;
    font-size: 13pt;
}}

.mark-symbol {{
    margin-left: 6px;
}}

.marking-line {{
    color: #d90000;
    font-family: Arial, sans-serif;
    font-weight: bold;
    font-size: 9.5pt;
    margin-top: 3px;
}}

.marking-explanation {{
    color: #d90000;
    font-family: Arial, sans-serif;
    font-size: 9.5pt;
    margin-bottom: 6px;
}}

.sub-score {{
    font-weight: bold;
    color: #d90000;
    font-size: 12pt;
    border: 1.5px solid #d90000;
    padding: 4px 8px;
    border-radius: 4px;
    display: inline-block;
    background-color: #fff;
}}


/* ========================================= */
/* CORRECT WORKING - GREEN */
/* ========================================= */

.correct-section {{
    background-color: #f2fff2;
    border-left: 3px solid #008000;
    padding: 8px 10px;
    margin-top: 12px;
    margin-bottom: 8px;
    border-radius: 4px;
}}

.correct-heading {{
    color: #008000;
    font-family: Arial, sans-serif;
    font-weight: bold;
    margin-bottom: 5px;
}}

.correct-line {{
    color: #008000;
    font-family: "Courier New", monospace;
    font-size: 10.5pt;
    margin-top: 4px;
    white-space: pre-wrap;
}}


/* ========================================= */
/* VISUALS */
/* ========================================= */

.visual-section {{
    margin-top: 12px;
}}

.math-table {{
    border-collapse: collapse;
    margin: 0 auto;
    color: #008000;
    font-family: Arial, sans-serif;
    font-size: 9.5pt;
}}

.math-table th {{
    border: 1px solid #008000;
    padding: 5px 8px;
    background-color: #f2fff2;
}}

.math-table td {{
    border: 1px solid #008000;
    padding: 5px 8px;
    text-align: center;
}}

.graph-image {{
    max-width: 100%;
    height: auto;
}}


/* ========================================= */
/* FEEDBACK */
/* ========================================= */

.feedback-box {{
    border: 2px solid #d90000;
    background-color: #fff0f0;
    border-radius: 6px;
    padding: 15px;
    margin-top: 20px;
    page-break-inside: avoid;
}}

.feedback-title {{
    color: #d90000;
    font-weight: bold;
    font-size: 12pt;
    margin-bottom: 8px;
    text-transform: uppercase;
}}

</style>

</head>


<body>


<div class="summary-header">

<h1>
Mathematics Script Examination Result
</h1>

<div class="summary-stats">

<span>
TOTAL SCORE:
{total_score}/{max_score}
</span>

|

<span>
PERCENTAGE:
{percentage}%
</span>

</div>

</div>


<div
style="
margin-bottom: 15px;
font-weight: bold;
color: #111;
"
>

{html.escape(
    str(
        student_data.get(
            "instruction",
            ""
        )
    )
)}

</div>


{questions_html}


<div class="feedback-box">

<div class="feedback-title">
Examiner's Remark &amp; Feedback
</div>

<div class="feedback-content">

<ul>

<li>

<strong>
Key Strengths:
</strong>

<ul>
{strengths_html}
</ul>

</li>


<li>

<strong>
Areas for Improvement:
</strong>

<ul>
{improvements_html}
</ul>

</li>

</ul>

</div>

</div>


</body>

</html>
"""

                # --------------------------------------------
                # GENERATE PDF
                # --------------------------------------------

                pdf_bytes = weasyprint.HTML(
                    string=html_content
                ).write_pdf()

                st.session_state.pdf_bytes = pdf_bytes

                st.success(
                    "Evaluation complete!"
                )

            except Exception as e:

                st.error(
                    f"Error processing scripts: {e}"
                )


# ============================================================
# DOWNLOAD BUTTON
# ============================================================

if st.session_state.pdf_bytes:

    st.download_button(
        label="📄 Download Combined Marked PDF",
        data=st.session_state.pdf_bytes,
        file_name="combined_marked_script.pdf",
        mime="application/pdf"
    )

