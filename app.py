import streamlit as st
import weasyprint
import json
import io
import re
import random
from PIL import Image
from google import genai
from pypdf import PdfReader
from pdf2image import convert_from_bytes

# Page configuration
st.set_page_config(page_title="AI Math Script Examiner", page_icon="📝", layout="centered")

st.title("📝 AI Math Script Examiner")
st.write("Upload handwritten student scripts (JPG, PNG, or PDF) to generate a single combined annotated PDF report.")

# Fetch API key(s) safely from Streamlit Secrets
raw_keys = st.secrets.get("GEMINI_API_KEYS") or st.secrets.get("GEMINI_API_KEY")

if not raw_keys:
    st.error("No API key found in Streamlit Secrets!")
    st.stop()

# Ensure api_keys_list is always a list of clean strings
if isinstance(raw_keys, list):
    api_keys_list = [str(k).strip() for k in raw_keys]
elif isinstance(raw_keys, str):
    # Handle comma-separated keys or single key
    api_keys_list = [k.strip() for k in raw_keys.split(",") if k.strip()]
else:
    api_keys_list = [str(raw_keys).strip()]

# Allow multiple files & multiple formats
uploaded_files = st.file_uploader(
    "Choose student script files (Images or PDFs)...", 
    type=["jpg", "jpeg", "png", "pdf"], 
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Grade All Scripts & Generate Combined PDF", type="primary"):
        with st.spinner("Processing files, evaluating handwriting, and compiling PDF..."):
            try:
                # Randomly select a key to distribute request limits across accounts
                selected_key = random.choice(api_keys_list)
                client = genai.Client(api_key=selected_key)
                
                all_images = []

                # 1. Convert all uploaded files/pages into PIL Images
                for file in uploaded_files:
                    file_bytes = file.read()
                    if file.name.lower().endswith('.pdf'):
                        # Convert PDF pages to images
                        pdf_images = convert_from_bytes(file_bytes)
                        all_images.extend(pdf_images)
                    else:
                        # Standard image file
                        img = Image.open(io.BytesIO(file_bytes))
                        all_images.append(img)

                st.info(f"Loaded {len(all_images)} total page(s) across {len(uploaded_files)} file(s). Evaluating...")

                prompt = """
                You are an expert secondary school mathematics teacher and national examiner. 
                Analyze the provided image(s) of handwritten student work.
                
                Evaluate every question step-by-step. For any step where the student made a mistake or left a step/question unattempted:
                1. Provide a clear, step-by-step concept breakdown explaining why the error occurred and how to solve it from first principles.
                2. FORMATTING FOR STEPS: When writing step-by-step explanations in "concept_explanation", DO NOT use plain numbers like "1)", "2)", "1.", or "2.". Instead, explicitly label each step using the exact format **[Step 1]**, **[Step 2]**, **[Step 3]** at the beginning of each step.
                3. VISUAL CORRECTIONS: If the question involves geometry (e.g., sectors, segments, angles), coordinate graphs (e.g., cubic curves, transformations, rotations), or data distribution (e.g., frequency tables, cumulative frequency):
                   - Include clean, well-formatted raw inline SVG code or raw HTML table code in "visual_correction" to visually demonstrate the correct answer.
                   - Keep SVG dimensions around width="350" height="200" with clear lines, labels, and stroke colors.
                
                Return ONLY a single valid JSON object with NO markdown formatting around it.
                Use this exact JSON format:
                {
                    "instruction": "Question heading/instructions from image(s)",
                    "questions": [
                        {
                            "title": "Question Title",
                            "max_score": 4,
                            "score": 2,
                            "learner_working": [
                                {
                                    "text": "Student working text", 
                                    "tick": false,
                                    "cross": true,
                                    "strikethrough": true, 
                                    "error_label": "Incorrect Graph / Calculation", 
                                    "correction": "Correct mathematical output",
                                    "concept_explanation": "**[Step 1]** First step detail... **[Step 2]** Second step detail...",
                                    "visual_correction": "<svg width='300' height='180'>...</svg>" 
                                }
                            ]
                        }
                    ],
                    "feedback": {
                        "strengths": ["Key strengths"],
                        "improvements": ["Areas for improvement"]
                    }
                }
                """

                # Send all collected page images to Gemini simultaneously
                contents_payload = all_images + [prompt]
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=contents_payload
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

                # 2. Render HTML & Generate Combined PDF
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
                        working_lines_html += f'<div style="margin-top: 6px;"><span{strikethrough_style}>{text}</span> {is_tick}{is_cross} {err_tag}</div>'
                        
                        if line.get("correction"):
                            working_lines_html += f'<div class="correction">Correct Answer: {line["correction"]}</div>'
                        
                        # Step-by-Step Concept Guide Box with Blue Step Highlights
                        if line.get("concept_explanation"):
                            explanation_text = line["concept_explanation"]
                            
                            # Format [Step X] tags into styled HTML spans
                            formatted_explanation = re.sub(
                                r'(\*\*\[Step \d+\]\*\*|\[Step \d+\]|\*\*Step \d+:\*\*)', 
                                r'<span class="step-tag">\1</span>', 
                                explanation_text
                            )

                            working_lines_html += f'''
                            <div class="explanation-box">
                                <strong>💡 Step-by-Step Concept Guide:</strong><br/>
                                {formatted_explanation}
                            </div>
                            '''

                        # Dynamic Visual Corrections (Tables, SVG Graphs, Geometry)
                        if line.get("visual_correction"):
                            working_lines_html += f'''
                            <div class="visual-box">
                                <strong>📊 Visual Correction & Illustration:</strong><br/>
                                <div class="visual-content">{line["visual_correction"]}</div>
                            </div>
                            '''

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
        .explanation-box {{ background-color: #fff9e6; border-left: 3px solid #ff9900; color: #333; font-size: 9.5pt; padding: 8px 12px; margin-top: 6px; margin-bottom: 8px; border-radius: 4px; line-height: 1.4; }}
        .step-tag {{ color: #0056b3; font-weight: bold; margin-right: 4px; display: inline-block; margin-top: 4px; }}
        .visual-box {{ background-color: #f0f8ff; border-left: 3px solid #0066cc; color: #222; font-size: 9.5pt; padding: 10px 12px; margin-top: 6px; margin-bottom: 8px; border-radius: 4px; }}
        .visual-content {{ margin-top: 8px; text-align: center; }}
        .visual-content table {{ margin: 0 auto; border-collapse: collapse; font-size: 9pt; }}
        .visual-content th, .visual-content td {{ border: 1px solid #0066cc; padding: 4px 8px; text-align: center; }}
        .visual-content th {{ background-color: #e6f2ff; }}
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

                st.success("All pages evaluated successfully!")
                st.download_button(
                    label="📄 Download Combined Marked PDF",
                    data=pdf_bytes,
                    file_name="combined_marked_script.pdf",
                    mime="application/pdf"
                )

            except Exception as e:
                st.error(f"Error processing scripts: {e}")
