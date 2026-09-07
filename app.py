
import streamlit as st
import weasyprint
import json
import io
import time
import pandas as pd
from PIL import Image
from google import genai
from google.genai import types
from pdf2image import convert_from_bytes

# Configure Streamlit Page
st.set_page_config(page_title="Universal AI Script Examiner", page_icon="📝", layout="centered")

st.title("📝 Universal AI Script Examiner")
st.write("Upload an **Official Marking Scheme** and **Student Scripts** (JPG, PNG, or PDF) for automated grading.")

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
# 2. RESILIENT API RETRY & FALLBACK ENGINE
# ==========================================
def generate_content_with_retry(client, contents_payload, config):
    candidate_models = ['gemini-3.6-flash', 'gemini-3.0-flash', 'gemini-2.5-flash']
    
    for model_name in candidate_models:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents_payload,
                    config=config
                )
                return response
            except Exception as e:
                err_msg = str(e)
                if "404" in err_msg or "NOT_FOUND" in err_msg:
                    break  # Skip deprecated models immediately
                if any(code in err_msg for code in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"]):
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise e
                    
    raise Exception("All API candidate models are currently unavailable. Please try again shortly.")

# ==========================================
# 3. OPTIMIZED IMAGE & PDF HELPER
# ==========================================
def process_and_compress_image(img):
    """Resizes and compresses PIL Image to JPEG bytes to minimize network latency."""
    img.thumbnail((800, 800))
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=70, optimize=True)
    buffer.seek(0)
    return types.Part.from_bytes(data=buffer.read(), mime_type="image/jpeg")

def extract_compressed_parts_from_files(uploaded_files):
    """Converts uploaded PDFs or Images into optimized API Image Parts directly."""
    image_parts = []
    for file in uploaded_files:
        file_bytes = file.read()
        if file.name.lower().endswith('.pdf'):
            pdf_images = convert_from_bytes(file_bytes, dpi=110)
            for img in pdf_images:
                image_parts.append(process_and_compress_image(img))
        else:
            img = Image.open(io.BytesIO(file_bytes))
            image_parts.append(process_and_compress_image(img))
    return image_parts

def generate_dynamic_table_html(table_data):
    if not table_data or "headers" not in table_data or "rows" not in table_data:
        return ""
    headers = table_data["headers"]
    rows = table_data["rows"]
    df = pd.DataFrame(rows, columns=headers)
    return df.to_html(index=False, classes="rendered-data-table")

# ==========================================
# 4. STREAMLIT INTERFACE
# ==========================================
col1, col2 = st.columns(2)

with col1:
    scheme_files = st.file_uploader(
        "1. Upload Marking Scheme", 
        type=["jpg", "jpeg", "png", "pdf"], 
        accept_multiple_files=True,
        key="scheme"
    )

with col2:
    script_files = st.file_uploader(
        "2. Upload Student Scripts", 
        type=["jpg", "jpeg", "png", "pdf"], 
        accept_multiple_files=True,
        key="scripts"
    )

enable_code_exec = st.checkbox(
    "Enable Code Execution (Turn on for Math/Physics/Accounting verification)", 
    value=False, 
    help="Disabling this speeds up evaluation for Humanities, Business Studies, and Languages."
)

if scheme_files and script_files:
    if st.button("Grade Scripts Against Marking Scheme", type="primary", use_container_width=True):
        with st.spinner("Compressing images, evaluating script, and generating PDF..."):
            try:
                scheme_parts = extract_compressed_parts_from_files(scheme_files)
                script_parts = extract_compressed_parts_from_files(script_files)

                num_scheme_imgs = len(scheme_parts)
                num_script_imgs = len(script_parts)

                st.info(f"Loaded {num_scheme_imgs} Marking Scheme page(s) and {num_script_imgs} Student Script page(s). Evaluating...")

                prompt = f"""
                You are a professional secondary school educational examiner capable of marking any subject (STEM, Humanities, Business Studies, Languages, Vocational).

                INPUT STRUCTURE:
                - The first {num_scheme_imgs} image(s) provided are the OFFICIAL MARKING SCHEME / ANSWER KEY.
                - The following {num_script_imgs} image(s) are the STUDENT'S HANDWRITTEN SCRIPT.

                EXAMINATION DIRECTIVES:
                1. STRICT SCHEME ALIGNMENT: Evaluate the student's work strictly against the provided Marking Scheme. Do not penalize for alternative phrasing in essay/concept questions if the underlying concept matches the scheme's criteria.
                2. MARKS ALLOCATION: Award partial and full marks strictly according to the mark breakdowns shown in the scheme.
                3. COMPUTATIONAL ACCURACY: For quantitative or accounting questions, verify calculations step-by-step.
                4. EXTRACT QUESTION STATEMENTS: Capture the question text or prompt in "question_text".

                Return a JSON object adhering strictly to this format:
                {{
                    "instruction": "Subject Name & Examination Header (e.g., Business Studies Paper 2 / Mathematics Mock Exam)",
                    "questions": [
                        {{
                            "title": "Question Identifier (e.g., Question 1a)",
                            "question_text": "Text statement of the question",
                            "max_score": 10,
                            "score": 7,
                            "needs_visual": null,
                            "table_data": null,
                            "working": [
                                {{
                                    "text": "Student point, statement, or calculation step",
                                    "correct": true,
                                    "error_type": "None / Incorrect Concept / Calculation Error / Missing Keyword / Unattempted",
                                    "correction": "Expected answer or keyword required by the scheme",
                                    "explanation": "Brief explanation of why marks were awarded or deducted based on the scheme."
                                }}
                            ]
                        }}
                    ],
                    "feedback": {{
                        "strengths": ["Key area where student demonstrated strong alignment with scheme"],
                        "improvements": ["Specific concept, method, or keyword missed according to scheme"]
                    }}
                }}
                """

                contents_payload = scheme_parts + script_parts + [prompt]
                
                tools = [{"code_execution": {}}] if enable_code_exec else []
                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                    tools=tools
                )

                response = generate_content_with_retry(client, contents_payload, config)

                student_data = json.loads(response.text.strip())

                total_score = sum(item.get("score", 0) for item in student_data.get("questions", []))
                max_score = sum(item.get("max_score", 0) for item in student_data.get("questions", []))
                percentage = round((total_score / max_score) * 100, 1) if max_score > 0 else 0

                # ==========================================
                # 5. HTML COMPOSITION ENGINE
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
                        if not is_correct and line.get("error_type") and line.get("error_type") != "None":
                            err_tag = f'<span class="err-label">{line["error_type"]}</span>'

                        strikethrough_style = ' style="text-decoration: line-through;"' if not is_correct else ""
                        working_lines_html += f'<div style="margin-top: 6px;"><span{strikethrough_style}>{text}</span> {tick_or_cross} {err_tag}</div>'
                        
                        if not is_correct and line.get("correction"):
                            working_lines_html += f'<div class="correction">Marking Guide Standard: {line["correction"]}</div>'
                        
                        if line.get("explanation"):
                            explanation_text = line["explanation"]
                            working_lines_html += f'''
                            <div class="explanation-box">
                                <span class="step-tag">[Point {step_idx}]</span> <strong>💡 Examiner Remarks:</strong><br/>
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
        .work-cell {{ width: 75%; font-family: Arial, sans-serif; font-size: 10pt; color: #002b80; background-color: #f8f9ff; border-left: 3px solid #002b80; padding: 10px; }}
        .marks-cell {{ width: 25%; text-align: right; padding-left: 10px; }}
        .tick {{ color: #008000; font-weight: bold; font-size: 12pt; }}
        .cross {{ color: #d90000; font-weight: bold; font-size: 12pt; }}
        .err-label {{ color: #d90000; font-weight: bold; font-size: 9pt; background-color: #ffe6e6; padding: 2px 6px; border-radius: 3px; display: inline-block; }}
        .correction {{ color: #d90000; font-weight: bold; font-size: 9.5pt; margin-top: 4px; }}
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
        <h1>Script Evaluation Report</h1>
        <div class="summary-stats">
            <span>TOTAL SCORE: {total_score}/{max_score}</span> | <span>PERCENTAGE: {percentage}%</span>
        </div>
    </div>
    <div style="margin-bottom: 15px; font-weight: bold; text-align: center; color: #333;">{student_data.get("instruction", "")}</div>
    {questions_html}
    <div class="feedback-box">
        <div class="feedback-title">Examiner's Remarks &amp; Feedback</div>
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
                    label="📄 Download Annotated PDF Report",
                    data=pdf_bytes,
                    file_name="evaluated_script.pdf",
                    mime="application/pdf"
                )

            except Exception as e:
                st.error(f"Error processing scripts: {e}")
