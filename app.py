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
    "to generate a marked mathematics report."
)


# ============================================================
# GEMINI API KEYS
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
        str(key).strip()
        for key in raw_keys
        if str(key).strip()
    ]

elif isinstance(raw_keys, str):
    api_keys_list = [
        key.strip()
        for key in raw_keys.split(",")
        if key.strip()
    ]

else:
    api_keys_list = [
        str(raw_keys).strip()
    ]


# ============================================================
# SESSION STATE
# ============================================================

if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

if "student_data" not in st.session_state:
    st.session_state.student_data = None


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_files = st.file_uploader(
    "Choose student script files (Images or PDFs)...",
    type=["jpg", "jpeg", "png", "pdf"],
    accept_multiple_files=True
)


# ============================================================
# HELPER: MATPLOTLIB IMAGE TO BASE64
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

    return (
        "data:image/png;base64,"
        + encoded
    )


# ============================================================
# HELPER: DRAW FUNCTION GRAPH
# ============================================================

def draw_function_graph(data):

    equation = str(
        data.get("equation", "")
    ).strip()

    if not equation:
        return ""

    try:

        expression = equation.replace(
            "^",
            "**"
        )

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

        ax.plot(
            x,
            y
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
# HELPER: DRAW POINT GRAPH
# ============================================================

def draw_points_graph(data):

    points = data.get(
        "points",
        []
    )

    if not points:
        return ""

    try:

        x_values = [
            float(point["x"])
            for point in points
        ]

        y_values = [
            float(point["y"])
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
# HELPER: DRAW GEOMETRY
# ============================================================

def draw_geometry(data):

    points = data.get(
        "points",
        []
    )

    if len(points) < 2:
        return ""

    try:

        x_values = [
            float(point["x"])
            for point in points
        ]

        y_values = [
            float(point["y"])
            for point in points
        ]

        if len(points) > 2:

            x_values.append(
                float(points[0]["x"])
            )

            y_values.append(
                float(points[0]["y"])
            )

        fig, ax = plt.subplots(
            figsize=(5, 4)
        )

        ax.plot(
            x_values,
            y_values,
            marker="o"
        )

        for point in points:

            label = point.get(
                "label",
                ""
            )

            ax.text(
                float(point["x"]),
                float(point["y"]),
                " " + str(label)
            )

        ax.set_aspect("equal")

        ax.grid(True)

        ax.set_title(
            data.get(
                "title",
                "Correct Diagram"
            )
        )

        return figure_to_base64(fig)

    except Exception:
        return ""


# ============================================================
# HELPER: CREATE TABLE
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

    table_html = (
        '<table class="math-table">'
    )

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
You are an expert secondary-school mathematics examiner.

Analyze the uploaded handwritten mathematics script carefully.

Your job is to identify every question and evaluate the student's
mathematical work fairly.

IMPORTANT:

1. Repeat the COMPLETE question in the "question_text" field.
   Repeating the question is essential.

2. Read and transcribe the student's working in the order written.

3. Mark each question fairly.

4. Award marks for correct mathematical method and correct answers.

5. Do not invent student work that is not visible.

6. If a student did not attempt a question, set:
   "status": "undone"

7. If a student attempted the question but made an error, set:
   "status": "incorrect"

8. If the student's entire solution is correct, set:
   "status": "correct"

9. ONLY provide "correct_solution" when:
   - status is "incorrect"
   OR
   - status is "undone"

10. If status is "correct", return:
    "correct_solution": []

11. For an undone question, provide the COMPLETE correct
    step-by-step mathematical working.

12. Keep explanations short and clear.

13. Do not explain correct student working.

VISUAL CORRECTIONS:

If no visual correction is needed:

"visual": null

For a table:

"visual": {
    "type": "table",
    "headers": ["Column 1", "Column 2"],
    "rows": [
        ["Value", "Value"]
    ]
}

For a function graph:

"visual": {
    "type": "function_graph",
    "title": "Correct Graph",
    "equation": "y = x^2 - 4",
    "x_min": -5,
    "x_max": 5
}

For a coordinate graph:

"visual": {
    "type": "points_graph",
    "title": "Correct Graph",
    "points": [
        {"x": -2, "y": 4},
        {"x": 0, "y": 0},
        {"x": 2, "y": 4}
    ]
}

For a simple geometry diagram:

"visual": {
    "type": "geometry",
    "title": "Correct Diagram",
    "points": [
        {"label": "A", "x": 0, "y": 0},
        {"label": "B", "x": 6, "y": 0},
        {"label": "C", "x": 3, "y": 4}
    ]
}

Return ONLY one valid JSON object.

Use exactly this structure:

{
    "instruction": "Exam heading or instruction",

    "questions": [
        {
            "title": "Question 1",

            "question_text": "Complete question",

            "max_score": 4,

            "score": 2,

            "status": "incorrect",

            "working": [
                {
                    "text": "Student written line",
                    "correct": true
                },
                {
                    "text": "Incorrect student line",
                    "correct": false,
                    "error_type": "Calculation Error",
                    "explanation": "Brief reason for the error."
                }
            ],

            "correct_solution": [
                "Correct mathematical step 1",
                "Correct mathematical step 2",
                "Correct final answer"
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
# PROCESS
# ============================================================

if uploaded_files:

    if st.button(
        "Grade All Scripts & Generate Marked PDF",
        type="primary"
    ):

        with st.spinner(
            "Reading and marking mathematics script..."
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

                    if file.name.lower().endswith(
                        ".pdf"
                    ):

                        pdf_images = (
                            convert_from_bytes(
                                file_bytes
                            )
                        )

                        for img in pdf_images:

                            img = img.convert("RGB")

                            img.thumbnail(
                                (1280, 1280)
                            )

                            all_images.append(img)

                    else:

                        img = Image.open(
                            io.BytesIO(
                                file_bytes
                            )
                        ).convert("RGB")

                        img.thumbnail(
                            (1280, 1280)
                        )

                        all_images.append(img)

                st.info(
                    f"Loaded {len(all_images)} page(s)."
                )

                # --------------------------------------------
                # GEMINI
                # --------------------------------------------

                contents_payload = (
                    all_images + [prompt]
                )

                response = (
                    client.models.generate_content(
                        model="gemini-3.6-flash",
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

                # --------------------------------------------
                # TOTAL MARKS
                # --------------------------------------------

                total_score = sum(
                    question.get(
                        "score",
                        0
                    )
                    for question in student_data.get(
                        "questions",
                        []
                    )
                )

                max_score = sum(
                    question.get(
                        "max_score",
                        0
                    )
                    for question in student_data.get(
                        "questions",
                        []
                    )
                )

                if max_score > 0:

                    percentage = round(
                        total_score
                        /
                        max_score
                        *
                        100,
                        1
                    )

                else:

                    percentage = 0

                # --------------------------------------------
                # BUILD QUESTIONS
                # --------------------------------------------

                questions_html = ""

                for question in student_data.get(
                    "questions",
                    []
                ):

                    question_title = html.escape(
                        str(
                            question.get(
                                "title",
                                "Question"
                            )
                        )
                    )

                    question_text = html.escape(
                        str(
                            question.get(
                                "question_text",
                                ""
                            )
                        )
                    )

                    score = question.get(
                        "score",
                        0
                    )

                    max_question_score = question.get(
                        "max_score",
                        0
                    )

                    status = question.get(
                        "status",
                        "incorrect"
                    )

                    # ----------------------------------------
                    # STUDENT WORK
                    # ----------------------------------------

                    working_html = ""

                    working = question.get(
                        "working",
                        []
                    )

                    if not working:

                        working_html = """
                        <div class="undone">
                            Not attempted.
                        </div>
                        """

                    else:

                        for line in working:

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
                                <div class="student-line">
                                    {text}
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

                                explanation = html.escape(
                                    str(
                                        line.get(
                                            "explanation",
                                            ""
                                        )
                                    )
                                )

                                working_html += f"""
                                <div class="student-line">
                                    {text}
                                </div>

                                <div class="marking-line">
                                    ✗ {error_type}
                                </div>

                                <div class="marking-explanation">
                                    {explanation}
                                </div>
                                """

                    # ----------------------------------------
                    # CORRECT SOLUTION
                    # ----------------------------------------

                    correct_solution_html = ""

                    correct_solution = question.get(
                        "correct_solution",
                        []
                    )

                    if status in [
                        "incorrect",
                        "undone"
                    ] and correct_solution:

                        correct_solution_html = """
                        <div class="correct-section">

                            <div class="correct-heading">
                                CORRECT WORKING
                            </div>
                        """

                        for step in correct_solution:

                            correct_solution_html += f"""
                            <div class="correct-line">
                                {html.escape(str(step))}
                            </div>
                            """

                        correct_solution_html += (
                            "</div>"
                        )

                    # ----------------------------------------
                    # VISUAL
                    # ----------------------------------------

                    visual_html = ""

                    visual = question.get(
                        "visual"
                    )

                    if visual:

                        visual_type = visual.get(
                            "type"
                        )

                        if visual_type == "table":

                            table = create_html_table(
                                visual
                            )

                            if table:

                                visual_html = f"""
                                <div class="visual-section">
                                    {table}
                                </div>
                                """

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
                                <div class="visual-section">
                                    <img
                                        src="{image_data}"
                                        class="graph-image"
                                    >
                                </div>
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
                                <div class="visual-section">
                                    <img
                                        src="{image_data}"
                                        class="graph-image"
                                    >
                                </div>
                                """

                        elif visual_type == (
                            "geometry"
                        ):

                            image_data = draw_geometry(
                                visual
                            )

                            if image_data:

                                visual_html = f"""
                                <div class="visual-section">
                                    <img
                                        src="{image_data}"
                                        class="graph-image"
                                    >
                                </div>
                                """

                    # ----------------------------------------
                    # QUESTION HTML
                    # ----------------------------------------

                    questions_html += f"""

                    <div class="question-block">

                        <div class="question-title">
                            {question_title}
                        </div>

                        <div class="question-text">
                            {question_text}
                        </div>

                        <div class="student-heading">
                            STUDENT'S WORK
                        </div>

                        <div class="student-work">
                            {working_html}
                        </div>

                        <div class="marking-score">
                            SCORE: {score}/{max_question_score}
                        </div>

                        {correct_solution_html}

                        {visual_html}

                    </div>
                    """

                # --------------------------------------------
                # FEEDBACK
                # --------------------------------------------

                feedback = student_data.get(
                    "feedback",
                    {}
                )

                strengths_html = ""

                for item in feedback.get(
                    "strengths",
                    []
                ):

                    strengths_html += (
                        "<li>"
                        + html.escape(
                            str(item)
                        )
                        + "</li>"
                    )

                improvements_html = ""

                for item in feedback.get(
                    "improvements",
                    []
                ):

                    improvements_html += (
                        "<li>"
                        + html.escape(
                            str(item)
                        )
                        + "</li>"
                    )

                # =================================================
                # HTML
                # =================================================

                html_content = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<style>

@page {{
    size: A4;
    margin: 15mm;
}}

body {{
    font-family: Arial, sans-serif;
    color: black;
    font-size: 11pt;
    line-height: 1.45;
}}

.summary {{
    border-bottom: 2px solid black;
    padding-bottom: 12px;
    margin-bottom: 18px;
}}

.summary h2 {{
    margin: 0 0 8px 0;
}}

.summary-stats {{
    font-weight: bold;
}}

.question-block {{
    margin-bottom: 25px;
    page-break-inside: avoid;
}}

.question-title {{
    color: black;
    font-weight: bold;
    font-size: 13pt;
    margin-bottom: 5px;
}}

.question-text {{
    color: black;
    font-weight: normal;
    white-space: pre-wrap;
    margin-bottom: 12px;
}}

.student-heading {{
    color: #0033cc;
    font-weight: bold;
    margin-top: 8px;
    margin-bottom: 4px;
}}

.student-work {{
    color: #0033cc;
    padding-left: 8px;
    border-left: 2px solid #0033cc;
}}

.student-line {{
    margin-top: 5px;
    white-space: pre-wrap;
}}

.undone {{
    font-style: italic;
}}

.marking-line {{
    color: #cc0000;
    font-weight: bold;
    margin-top: 4px;
}}

.marking-explanation {{
    color: #cc0000;
    font-size: 10pt;
    margin-bottom: 5px;
}}

.marking-score {{
    color: #cc0000;
    font-weight: bold;
    margin-top: 10px;
}}

.correct-section {{
    color: #008000;
    margin-top: 12px;
    padding-left: 8px;
    border-left: 3px solid #008000;
}}

.correct-heading {{
    color: #008000;
    font-weight: bold;
    margin-bottom: 5px;
}}

.correct-line {{
    color: #008000;
    margin-top: 4px;
    white-space: pre-wrap;
}}

.visual-section {{
    margin-top: 12px;
}}

.math-table {{
    border-collapse: collapse;
    margin-top: 10px;
}}

.math-table th,
.math-table td {{
    border: 1px solid black;
    padding: 6px 10px;
    text-align: center;
}}

.graph-image {{
    max-width: 100%;
    height: auto;
}}

.feedback {{
    border-top: 2px solid black;
    margin-top: 25px;
    padding-top: 12px;
}}

.feedback h3 {{
    margin-top: 0;
}}

</style>

</head>

<body>

<div class="summary">

<h2>
Mathematics Script Examination Result
</h2>

<div class="summary-stats">

TOTAL SCORE:
{total_score}/{max_score}

&nbsp;&nbsp;&nbsp;

PERCENTAGE:
{percentage}%

</div>

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

<br>

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

                # --------------------------------------------
                # CREATE PDF
                # --------------------------------------------

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
# DOWNLOAD BUTTON
# ============================================================

if st.session_state.pdf_bytes:

    st.download_button(
        label="📄 Download Marked PDF",
        data=st.session_state.pdf_bytes,
        file_name="marked_math_script.pdf",
        mime="application/pdf"
    )

