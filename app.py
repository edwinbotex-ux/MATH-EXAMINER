import streamlit as st
import weasyprint
import json
import io
import re
import base64
import matplotlib.pyplot as plt
from PIL import Image
from google import genai
from pdf2image import convert_from_bytes

# Configure Page
st.set_page_config(page_title="AI Math Script Examiner", page_icon="📝", layout="centered")

st.title("📝 AI Math Script Examiner")
st.write("Upload handwritten student scripts (JPG, PNG, or PDF) to generate an annotated PDF report.")

# ==========================================
# 1. API KEY SETUP & SAFE HANDLING
# ==========================================
raw_keys = st.secrets.get("GEMINI_API_KEYS") or st.secrets.get("GEMINI_API_KEY")

if not raw_keys:
    st.error("No API key found in Streamlit Secrets!")
    st.stop()

# Normalize keys into a list
if isinstance(raw_keys, list):
    api_keys_list = [str(k).strip() for k in raw_keys]
elif isinstance(raw_keys, str):
    api_keys_list = [k.strip() for k in raw_keys.split(",") if k.strip()]
else:
    api_keys_list = [str(raw_keys).strip()]

# Round-robin key selection via session_state to prevent hitting single-key limits
if "key_index" not in st.session_state:
    st.session_state.key_index = 0

selected_key = api_keys_list[st.session_state.key_index % len(api_keys_list)]
st.session_state.key_index += 1

client = genai.Client(api_key=selected_key)

# ==========================================
# 2. HELPER FUNCTIONS FOR VISUALS
# ==========================================
def generate_math_graph_base64(x_vals, y_vals, title="Graph Verification", xlabel="x", ylabel="y"):
    """
    Generates a clean mathematical plot using Matplotlib and encodes 
    it as a Base64 SVG data URI for direct insertion into WeasyPrint.
    """
    fig, ax = plt.subplots(figsize=(4, 2.5), dpi=150)
    ax.plot(x_vals, y_vals, color="#0056b3", linewidth=2, marker='o', markersize=4)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    buf = io.BytesIO()
    plt.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    
    svg_str = buf.getvalue().decode("utf-8")
    b64_svg = base64.b64encode(svg_str.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;charset=utf-8;base64,{b64_svg}"

# ==========================================
# 3. STREAMLIT INTERFACE & FILE UPLOADER
# ==========================================
uploaded_files = st.file_uploader(
    "Choose student script files (Images or PDFs)...", 
    type=["jpg", "jpeg", "png", "pdf"], 
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Grade All Scripts & Generate Combined PDF", type="primary"):
        with st.spinner("Processing files, evaluating handwriting, and compiling PDF..."):
            try:
                all_images = []

                # Convert uploaded files into PIL Images (Downscaling reduces vision token costs by ~70%)
                for file in uploaded_files:
                    file_bytes = file.read()
                    if file.name.lower().endswith('.pdf'):
                        pdf_images = convert_from_bytes(file_bytes)
                        for img in pdf_images:
                            img.thumbnail((1280, 1280))
                            all_images.append(img)
                    else:
                        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                        img.thumbnail((1280, 1280))
                        all_images.append(img)

                st.info(f"Loaded {len(all_images)} script page(s). Requesting evaluation...")

                # Lightweight Prompt: Only asks Gemini for logic evaluation and raw JSON
                prompt = """
                You are a math examiner. Analyze the uploaded student script.
                Return ONLY a valid JSON object:
                {
                    "instruction": "Exam heading/title",
                    "questions": [
                        {
                            "title": "Question 1",
                            "max_score": 4,
                            "score": 2,
                            "working": [
                                {
                                    "text": "Student written line",
                                    "correct": false,
                                    "error_type": "Calculation Error",
                                    "correction": "Correct line output",
                                    "explanation": "Brief 1-sentence reason for error"
                                }
                            ]
                        }
                    ],
                    "feedback": {
                        "strengths": ["1-2 key strengths"],
                        "improvements": ["1-2 key improvements"]
                    }
                }
                """

                contents_payload = all_images + [prompt]
                response = client.models.generate_content(
                    model='gemini-3.6-flash',
                    contents=contents_payload
                )

                # Clean response string to parse JSON safely
                raw_text = response.text.strip()
                if raw_text.startswith("```json"):
                    raw_text = raw_text[7:]
                if raw_text.startswith("```"):
                    raw_text = raw_text[3:]
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3]

                student_data = json.loads(raw_text.strip())

                # Compute totals
                total_score = sum(item.get("score", 0) for item in student_data.get("questions", []))
                max_score = sum(item.get("max_score", 0) for item in student_data.get("questions", []))
                percentage = round((total_score / max_score) * 100, 1) if max_score > 0 else 0

                # ==========================================
                # 4. PYTHON RENDERING ENGINE (HTML/CSS)
                # ==========================================
                questions_html = ""
                for q in student_data.get("questions", []):
                    working_lines_html = ""
                    
                    # Robust key fallbacks for step evaluation array
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
                        
                        # Injects step badge tags programmatically
                        if not is_correct and line.get("explanation"):
                            explanation_text = line["explanation"]
                            working_lines_html += f'''
                            <div class="explanation-box">
                                <span class="step-tag">[Step {step_idx}]</span> <strong>💡 Concept Guide:</strong><br/>
                                {explanation_text}
                            </div>
                            '''

                    # Optional: Render Python chart dynamically if tabular/coordinate reference exists
                    graph_html = ""
                    if "graph" in q.get("title", "").lower() or "plot" in q.get("title", "").lower():
                        sample_b64 = generate_math_graph_base64([-2, -1, 0, 1, 2], [4, 1, 0, 1, 4], title="Correct Plot Reference: y = x^2")
                        graph_html = f'<div class="graph-container"><img src="{sample_b64}" /></div>'

                    questions_html += f"""
                    <div class="question-block">
                        <div class="question-title">{q.get("title", "Question")} &nbsp;&nbsp;&nbsp; [{q.get("max_score", 0)} Marks]</div>
                        <table class="script-table">
                            <tr>
                                <td class="script-cell work-cell">
                                    {working_lines_html}
                                    {graph_html}
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

                # Final Template Construction
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
        .tick {{ color: #008000; font-weight: bold; font-size: 12pt; }}
        .cross {{ color: #d90000; font-weight: bold; font-size: 12pt; }}
        .err-label {{ color: #d90000; font-weight: bold; font-size: 9.5pt; background-color: #ffe6e6; padding: 2px 6px; border-radius: 3px; display: inline-block; }}
        .correction {{ color: #d90000; font-weight: bold; font-family: "Courier New", monospace; font-size: 10pt; margin-top: 4px; }}
        .explanation-box {{ background-color: #fff9e6; border-left: 3px solid #ff9900; color: #333; font-size: 9.5pt; padding: 8px 12px; margin-top: 6px; margin-bottom: 8px; border-radius: 4px; line-height: 1.4; }}
        .step-tag {{ color: #0056b3; font-weight: bold; margin-right: 4px; display: inline-block; }}
        .sub-score {{ font-weight: bold; color: #d90000; font-size: 12pt; border: 1.5px solid #d90000; padding: 4px 8px; border-radius: 4px; display: inline-block; background-color: #fff; }}
        .feedback-box {{ border: 2px solid #d90000; background-color: #fff0f0; border-radius: 6px; padding: 15px; margin-top: 20px; page-break-inside: avoid; }}
        .feedback-title {{ color: #d90000; font-weight: bold; font-size: 12pt; margin-bottom: 8px; text-transform: uppercase; }}
        .graph-container {{ text-align: center; margin: 10px 0; }}
        .graph-container img {{ max-width: 90%; height: auto; border: 1px solid #ccc; border-radius: 4px; padding: 4px; background: #fff; }}
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
                # Generate PDF using WeasyPrint
                pdf_bytes = weasyprint.HTML(string=html_content).write_pdf()

                st.success("Evaluation complete!")
                st.download_button(
                    label="📄 Download Combined Marked PDF",
                    data=pdf_bytes,
                    file_name="combined_marked_script.pdf",
                    mime="application/pdf"
                )

            except Exception as e:
                st.error(f"Error processing scripts: {e}")
