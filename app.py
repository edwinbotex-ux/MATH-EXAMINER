import streamlit as st
import weasyprint
import json
import io
import pandas as pd
from PIL import Image
from google import genai
from pdf2image import convert_from_bytes

# Configure Streamlit Page
st.set_page_config(page_title="AI Math Script Examiner", page_icon="📝", layout="centered")

st.title("📝 AI Math Script Examiner")
st.write("Upload handwritten student scripts (JPG, PNG, or PDF) to generate an annotated PDF report.")

# ==========================================
# 1. API KEY SETUP
# ==========================================
raw_keys = st.secrets.get("GEMINI_API_KEYS") or st.secrets.get("GEMINI_API_KEY")

if not raw_keys:
    st.error("No API key found in Streamlit Secrets!")
    st.stop()

if isinstance(raw_keys, list):
    api_keys_list = [str(k).strip() for k in raw_keys]
elif isinstance(raw_keys, str):
    api_keys_list = [k.strip() for k in raw_keys.split(",") if k.strip()]
else:
    api_keys_list = [str(raw_keys).strip()]

if "key_index" not in st.session_state:
    st.session_state.key_index = 0

selected_key = api_keys_list[st.session_state.key_index % len(api_keys_list)]
st.session_state.key_index += 1

client = genai.Client(api_key=selected_key)

# ==========================================
# 2. DYNAMIC TABLE RENDERER
# ==========================================
def generate_dynamic_table_html(table_data):
    if not table_data or "headers" not in table_data or "rows" not in table_data:
        return ""
    
    headers = table_data["headers"]
    rows = table_data["rows"]
    
    df = pd.DataFrame(rows, columns=headers)
    return df.to_html(index=False, classes="rendered-data-table")

# ==========================================
# 3. STREAMLIT INTERFACE & FILE PROCESSING
# ==========================================
uploaded_files = st.file_uploader(
    "Choose student script files (Images or PDFs)...", 
    type=["jpg", "jpeg", "png", "pdf"], 
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Grade All Scripts & Generate Combined PDF", type="primary"):
        with st.spinner("Evaluating handwriting and compiling PDF..."):
            try:
                all_images = []

                for file in uploaded_files:
                    file_bytes = file.read()
                    if file.name.lower().endswith('.pdf'):
                        pdf_images = convert_from_bytes(file_bytes)
                        for img in pdf_images:
                            img.thumbnail((1024, 1024)) # Reduced slightly for faster upload/processing
                            all_images.append(img)
                    else:
                        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                        img.thumbnail((1024, 1024))
                        all_images.append(img)

                st.info(f"Loaded {len(all_images)} script page(s). Evaluating...")

                prompt = """
                You are a secondary school mathematics examiner evaluating handwritten student scripts.

                CRITICAL MATHEMATICAL INTEGRITY DIRECTIVES:
                1. BEFORE outputting the JSON, independently calculate every math solution or equation root step-by-step using exact analytical algebra.
                2. Do NOT approximate or guess numerical answers. Always compute exact mathematical roots.
                3. Extract the full written question statement into "question_text".

                Return ONLY a valid JSON object matching this schema exactly (do NOT include plot/graph data):
                {
                    "instruction": "Exam heading/title",
                    "questions": [
                        {
                            "title": "Question number (e.g. Question 23)",
                            "question_text": "Full text statement of the question",
                            "max_score": 2,
                            "score": 0,
                            "needs_visual": "data_table", 
                            "table_data": {
                                "headers": ["x", "y1", "y2"],
                                "rows": [["0°", "1.00", "1.73"]]
                            },
                            "working": [
                                {
                                    "text": "Student response or Unattempted",
                                    "correct": false,
                                    "error_type": "Omission Error",
                                    "correction": "Exact calculated answer",
                                    "explanation": "Clear step-by-step reason explaining derivation."
                                }
                            ]
                        }
                    ],
                    "feedback": {
                        "strengths": ["Key strength"],
                        "improvements": ["Area for improvement"]
                    }
                }

                Note:
                - "needs_visual": Set to "data_table" ONLY if a table is explicitly required, otherwise null.
                """

                contents_payload = all_images + [prompt]
                
                response = client.models.generate_content(
                    model='gemini-3.6-flash',
                    contents=contents_payload,
                    config={
                        "tools": [{"code_execution": {}}]
                    }
                )

                raw_text = response.text.strip()
                if raw_text.startswith("```json"):
                    raw_text = raw_text[7:]
                if raw_text.startswith("```"):
                    raw_text = raw_text[3:]
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3]

                student_data = json.loads(raw_text.strip())

                total_score = sum(item.get("score", 0) for item in student_data.get("questions", []))
                max_score = sum(item.get("max_score", 0) for item in student_data.get("questions", []))
                percentage = round((total_score / max_score) * 100, 1) if max_score > 0 else 0

                # ==========================================
                # 4. HTML COMPOSITION ENGINE
                # ==========================================
                questions_html = ""
                for q in student_data.get("questions", []):
                    question_text = q.get("question_text", "")
                    working_lines_html = ""
                    working_list = q.get("working") or q.get("learner_working") or []
                    
                    for step_idx, line in enumerate(working_list, start=1):
                        text = line.get("text", "")
                        is_correct = line.get("correct", False)
                        
                        tick_or_cross = '<span class="tick">&#10003;</span>' if is_correct else '<span class="cross">&#10007;</span>'
                        
                        err_tag = ""
                        if not is_correct and line.get("error_type"):
                            err_tag = f'<span class="err-label">{line["error_type"]}</span>'

                        strikethrough_style = ' style="text-decoration: line-through;"' if not is_correct else ""
                        working_lines_html += f'<div style="margin-top: 6px;"><span{strikethrough_style}>{text}</span> {tick_or_cross} {err_tag}</div>'
                        
                        if not is_correct and line.get("correction"):
                            working_lines_html += f'<div class="correction">Correct Answer: {line["correction"]}</div>'
                        
                        if not is_correct and line.get("explanation"):
                            explanation_text = line["explanation"]
                            working_lines_html += f'''
                            <div class="explanation-box">
                                <span class="step-tag">[Step {step_idx}]</span> <strong>💡 Concept Guide:</strong><br/>
                                {explanation_text}
                            </div>
                            '''

                    visual_type = q.get("needs_visual")
                    visual_html = ""

                    if visual_type == "data_table":
                        tbl_html = generate_dynamic_table_html(q.get("table_data"))
                        visual_html = f'<div class="table-container" style="margin-top:10px;">{tbl_html}</div>'

                    questions_html += f"""
                    <div class="question-block">
                        <div class="question-title">{q.get("title", "Question")} &nbsp;&nbsp;&nbsp; [{q.get("max_score", 0)} Marks]</div>
                        {f'<div class="question-statement" style="font-weight: bold; margin-bottom: 8px; color: #333;">{question_text}</div>' if question_text else ''}
                        <table class="script-table">
                            <tr>
                                <td class="script-cell work-cell">
                                    {working_lines_html}
                                    {visual_html}
                                </td>
                                <td class="script-cell marks-cell">
                                    <div class="sub-score">{q.get("score", 0)} / {q.get("max_score", 0)}</div>
                                </td>
                            </tr>
                        </table>
                    </div>
                    """

                feedback = student_data.get("feedback", {})
                strengths_html = "".join([f"<li>{item}</li>" for item in feedback.get("strengths", [])])
                improvements_html = "".join([f"<li>{item}</li>" for item in feedback.get("improvements", [])])

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
        .question-title {{ font-size: 11pt; font-weight: bold; margin-bottom: 6px; border-bottom: 1px dashed #ccc; padding-bottom: 5px; }}
        .script-table {{ width: 100%; border-collapse: collapse; }}
        .script-cell {{ vertical-align: top; padding: 4px; }}
        .work-cell {{ width: 75%; font-family: "Courier New", monospace; font-size: 10.5pt; color: #002b80; background-color: #f8f9ff; border-left: 3px solid #002b80; padding: 10px; }}
        .marks-cell {{ width: 25%; text-align: right; padding-left: 10px; }}
        .tick {{ color: #008000; font-weight: bold; font-size: 12pt; }}
        .cross {{ color: #d90000; font-weight: bold; font-size: 12pt; }}
        .err-label {{ color: #d90000; font-weight: bold; font-size: 9.5pt; background-color: #ffe6e6; padding: 2px 6px; border-radius: 3px; display: inline-block; }}
        .correction {{ color: #d90000; font-weight: bold; font-family: "Courier New", monospace; font-size: 10pt; margin-top: 4px; }}
        .explanation-box {{ background-color: #fff9e6; border-left: 3px solid #ff9900; color: #333; font-size: 9.5pt; padding: 8px 12px; margin-top: 6px; margin-bottom: 8px; border-radius: 4px; line-height: 1.4; }}
        .step-tag {{ color: #0056b3; font-weight: bold; margin-right: 4px; display: inline-block; }}
        .sub-score {{ font-weight: bold; color: #d90000; font-size: 12pt; border: 1.5px solid #d90000; padding: 4px 8px; border-radius: 4px; display: inline-block; background-color: #fff; }}
        .feedback-box {{ border: 2px solid #d90000; background-color: #fff0f0; border-radius: 6px; padding: 15px; margin-top: 20px; page-break-inside: avoid; }}
        .feedback-title {{ color: #d90000; font-weight: bold; font-size: 12pt; margin-bottom: 8px; text-transform: uppercase; }}
        .rendered-data-table {{ width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 9.5pt; }}
        .rendered-data-table th, .rendered-data-table td {{ border: 1px solid #0056b3; padding: 5px 8px; text-align: center; }}
        .rendered-data-table th {{ background-color: #e6f2ff; color: #002b80; }}
    </style>
</head>
<body>
    <div class="summary-header">
        <h1>Mathematics Script Examination Result</h1>
        <div class="summary-stats">
            <span>TOTAL SCORE: {total_score}/{max_score}</span> | <span>PERCENTAGE: {percentage}%</span>
        </div>
    </div>
    <div style="margin-bottom: 15px; font-weight: bold;">{student_data.get("instruction", "")}</div>
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
                    file_name="marked_script.pdf",
                    mime="application/pdf"
                )

            except Exception as e:
                st.error(f"Error processing scripts: {e}")
