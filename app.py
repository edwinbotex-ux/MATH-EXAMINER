import streamlit as st
import weasyprint
import json
from PIL import Image
from google import genai

# Page configuration
st.set_page_config(page_title="AI Math Script Examiner", page_icon="📝", layout="centered")

st.title("📝 AI Math Script Examiner")
st.write("Upload a photograph of a student's handwritten math work to generate an annotated PDF report.")

# Sidebar API Key input
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

uploaded_file = st.file_uploader("Choose a student script image...", type=["jpg", "jpeg", "png"])

if uploaded_file and api_key:
    if st.button("Grade Script & Generate PDF", type="primary"):
        with st.spinner("Analyzing handwriting and generating PDF..."):
            try:
                # 1. Initialize Gemini Client
                client = genai.Client(api_key=api_key)
                img = Image.open(uploaded_file)

                prompt = """
                You are an expert mathematics examiner. 
                Analyze the provided image of handwritten student work.
                
                Evaluate every question step-by-step and return ONLY a single valid JSON object with NO markdown formatting around it.
                Use this exact JSON format:
                {
                    "instruction": "Question heading/instructions from image",
                    "questions": [
                        {
                            "title": "a) 3x² + 4x + 1 = 0",
                            "max_score": 3,
                            "score": 2,
                            "learner_working": [
                                {"text": "Sum = 4 { 1 and 3 }", "tick": true, "method_label": "M1 (Setup)"},
                                {"text": "x = -1/3 or 1", "strikethrough": true, "cross": true, "error_label": "Transcription error", "correction": "x = -1/3 or x = -1"}
                            ]
                        }
                    ],
                    "feedback": {
                        "strengths": ["List key strengths observed"],
                        "improvements": ["List specific areas for improvement"]
                    }
                }
                """

                response = client.models.generate_content(
                    model='gemini-3.6-flash',
                    contents=[img, prompt]
                )

                # Clean response string
                raw_text = response.text.strip()
                if raw_text.startswith("```json"):
                    raw_text = raw_text[7:]
                if raw_text.startswith("```"):
                    raw_text = raw_text[3:]
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3]

                student_data = json.loads(raw_text.strip())

                # 2. Render HTML & Generate PDF
                total_score = sum(item["score"] for item in student_data["questions"])
                max_score = sum(item["max_score"] for item in student_data["questions"])
                percentage = round((total_score / max_score) * 100, 1) if max_score > 0 else 0

                questions_html = ""
                for q in student_data["questions"]:
                    working_lines_html = ""
                    for line in q["learner_working"]:
                        text = line.get("text", "")
                        is_tick = '<span class="tick">&#10003;</span>' if line.get("tick") else ""
                        is_cross = '<span class="cross">&#10007;</span>' if line.get("cross") else ""
                        
                        err_tag = ""
                        if line.get("error_label"):
                            err_tag = f'<span class="err-label">{line["error_label"]}</span>'
                        elif line.get("method_label"):
                            err_tag = f'<span class="err-label" style="color:#008000; background:#e6ffe6;">{line["method_label"]}</span>'
                        
                        strikethrough_style = ' style="text-decoration: line-through;"' if line.get("strikethrough") else ""
                        working_lines_html += f'<div><span{strikethrough_style}>{text}</span> {is_tick}{is_cross} {err_tag}</div>'
                        
                        if line.get("correction"):
                            working_lines_html += f'<div class="correction">Correction: {line["correction"]}</div>'

                    questions_html += f"""
                    <div class="question-block">
                        <div class="question-title">{q["title"]} &nbsp;&nbsp;&nbsp; [{q["max_score"]} Marks]</div>
                        <table class="script-table">
                            <tr>
                                <td class="script-cell work-cell">{working_lines_html}</td>
                                <td class="script-cell marks-cell"><div class="sub-score">{q["score"]} / {q["max_score"]}</div></td>
                            </tr>
                        </table>
                    </div>
                    """

                strengths_html = "".join([f"<li>{item}</li>" for item in student_data["feedback"]["strengths"]])
                improvements_html = "".join([f"<li>{item}</li>" for item in student_data["feedback"]["improvements"]])

                html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <style>
        @page {{ size: A4; margin: 15mm; background-color: #fcfcfc; }}
        body {{ font-family: Arial, sans-serif; color: #111; font-size: 11pt; line-height: 1.5; }}
        .summary-header {{ border: 2px solid #d90000; background-color: #fff0f0; border-radius: 6px; padding: 12px 20px; margin-bottom: 20px; text-align: center; }}
        .summary-header h1 {{ color: #d90000; margin: 0 0 8px 0; font-size: 16pt; text-transform: uppercase; }}
        .summary-stats {{ font-size: 14pt; font-weight: bold; color: #b30000; }}
        .summary-stats span {{ margin: 0 15px; }}
        .question-block {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 15px; margin-bottom: 20px; }}
        .question-title {{ font-size: 11pt; font-weight: bold; margin-bottom: 10px; border-bottom: 1px dashed #ccc; padding-bottom: 5px; }}
        .script-table {{ width: 100%; border-collapse: collapse; }}
        .script-cell {{ vertical-align: top; padding: 4px; }}
        .work-cell {{ width: 75%; font-family: "Courier New", monospace; font-size: 10.5pt; color: #002b80; background-color: #f8f9ff; border-left: 3px solid #002b80; padding: 10px; }}
        .marks-cell {{ width: 25%; text-align: right; padding-left: 10px; }}
        .tick {{ color: #d90000; font-weight: bold; font-size: 12pt; }}
        .cross {{ color: #d90000; font-weight: bold; font-size: 12pt; }}
        .err-label {{ color: #d90000; font-weight: bold; font-size: 9.5pt; background-color: #ffe6e6; padding: 2px 6px; border-radius: 3px; display: inline-block; }}
        .correction {{ color: #d90000; font-weight: bold; font-family: "Courier New", monospace; font-size: 10pt; margin-top: 4px; }}
        .sub-score {{ font-weight: bold; color: #d90000; font-size: 12pt; border: 1.5px solid #d90000; padding: 4px 8px; border-radius: 4px; display: inline-block; background-color: #fff; }}
        .feedback-box {{ border: 2px solid #d90000; background-color: #fff0f0; border-radius: 6px; padding: 15px; margin-top: 20px; page-break-inside: avoid; }}
        .feedback-title {{ color: #d90000; font-weight: bold; font-size: 12pt; margin-bottom: 8px; text-transform: uppercase; }}
    </style>
</head>
<body>
    <div class="summary-header">
        <h1>Mathematics Script Examination Result</h1>
        <div class="summary-stats">
            <span>TOTAL SCORE: {total_score}/{max_score}</span> | <span>PERCENTAGE: {percentage}%</span>
        </div>
    </div>
    <div style="margin-bottom: 15px; font-weight: bold;">{student_data["instruction"]}</div>
    {questions_html}
    <div class="feedback-box">
        <div class="feedback-title">Examiner's Remark &amp; Feedback</div>
        <div class="feedback-content">
            <ul>
                <li><strong>Key Strengths:</strong><ul>{strengths_html}</ul></li>
                <li><strong>Areas for Improvement:</strong><ul>{improvements_html}</ul></li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
                pdf_bytes = weasyprint.HTML(string=html_content).write_pdf()

                st.success("Evaluation complete!")
                st.download_button(
                    label="📄 Download Marked PDF",
                    data=pdf_bytes,
                    file_name="marked_learner_script.pdf",
                    mime="application/pdf"
                )

            except Exception as e:
                st.error(f"Error processing script: {e}")
elif uploaded_file and not api_key:
    st.warning("Please enter your Gemini API Key in the left sidebar to proceed.")
