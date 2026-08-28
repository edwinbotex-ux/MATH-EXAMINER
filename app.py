import streamlit as st
import weasyprint
import json
import io
import re
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from PIL import Image
from google import genai
from pdf2image import convert_from_bytes

# Configure Streamlit Page
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

# Round-robin key selection via session_state to balance API quota usage
if "key_index" not in st.session_state:
    st.session_state.key_index = 0

selected_key = api_keys_list[st.session_state.key_index % len(api_keys_list)]
st.session_state.key_index += 1

client = genai.Client(api_key=selected_key)

# ==========================================
# 2. PYTHON VISUAL RENDERERS (Matplotlib & Pandas)
# ==========================================
def generate_function_graph_b64():
    """Generates Quadratic and Cubic Function Plot"""
    fig, ax = plt.subplots(figsize=(4.5, 3), dpi=150)
    x = np.linspace(-3, 3, 200)
    y_quad = x**2 - 2
    y_cubic = x**3 - 2*x

    ax.plot(x, y_quad, color='#0056b3', label=r'$y = x^2 - 2$')
    ax.plot(x, y_cubic, color='#d90000', label=r'$y = x^3 - 2x$')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title('Function Verification Plot', fontsize=9, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=8)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='svg', bbox_inches='tight')
    plt.close(fig)
    return f"data:image/svg+xml;charset=utf-8;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

def generate_transformation_graph_b64(angle_deg=90):
    """Generates Geometric Transformation Plot"""
    fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
    orig_tri = np.array([[1, 1], [3, 1], [1.5, 2.5], [1, 1]])

    theta = np.radians(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rot_tri = (R @ orig_tri.T).T

    ax.plot(orig_tri[:, 0], orig_tri[:, 1], 'b-o', label='Original')
    ax.fill(orig_tri[:, 0], orig_tri[:, 1], 'blue', alpha=0.2)
    ax.plot(rot_tri[:, 0], rot_tri[:, 1], 'r--o', label=f"Rotated {angle_deg}° CCW")
    ax.fill(rot_tri[:, 0], rot_tri[:, 1], 'red', alpha=0.2)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-2, 4)
    ax.set_aspect('equal')
    ax.set_title(f'Shape Rotation ({angle_deg}° CCW)', fontsize=9, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=8)

    buf = io.BytesIO()
    plt.savefig(buf, format='svg', bbox_inches='tight')
    plt.close(fig)
    return f"data:image/svg+xml;charset=utf-8;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

def generate_network_diagram_b64():
    """Generates Workflow / Network Diagram"""
    fig, ax = plt.subplots(figsize=(4.5, 2.5), dpi=150)
    G = nx.DiGraph()
    G.add_edges_from([('Input', 'Process'), ('Process', 'Option A'), 
                      ('Process', 'Option B'), ('Option A', 'Output'), ('Option B', 'Output')])
    pos = {'Input': (0, 1), 'Process': (1, 1), 'Option A': (2, 1.5), 'Option B': (2, 0.5), 'Output': (3, 1)}

    nx.draw(G, pos, ax=ax, with_labels=True, node_color='#e6f2ff', edge_color='#0056b3',
            node_size=1600, font_size=8, font_weight='bold', arrowsize=12)
    ax.set_title('Workflow / Network Model', fontsize=9, fontweight='bold')

    buf = io.BytesIO()
    plt.savefig(buf, format='svg', bbox_inches='tight')
    plt.close(fig)
    return f"data:image/svg+xml;charset=utf-8;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

def generate_geometry_patches_b64():
    """Generates Inscribed Geometry Model"""
    fig, ax = plt.subplots(figsize=(3, 3), dpi=150)
    circle = patches.Circle((0.5, 0.5), 0.35, edgecolor='#800080', facecolor='#f3e6ff', linewidth=2)
    rect = patches.Rectangle((0.2, 0.2), 0.6, 0.6, edgecolor='#006600', facecolor='none', linewidth=2, linestyle='--')

    ax.add_patch(circle)
    ax.add_patch(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Inscribed Geometry Model', fontsize=9, fontweight='bold')

    buf = io.BytesIO()
    plt.savefig(buf, format='svg', bbox_inches='tight')
    plt.close(fig)
    return f"data:image/svg+xml;charset=utf-8;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

def generate_data_table_html(x_vals=None):
    """Generates Clean HTML Data Table via Pandas"""
    if x_vals is None:
        x_vals = np.array([-2, -1, 0, 1, 2])
    
    table_data = {
        "x": x_vals,
        "f(x) = x² - 2": x_vals**2 - 2,
        "g(x) = x³ - 2x": x_vals**3 - 2*x_vals
    }
    df = pd.DataFrame(table_data)
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
        with st.spinner("Processing files, evaluating handwriting, and compiling PDF..."):
            try:
                all_images = []

                # Convert uploaded files into PIL Images (Downscaling saves ~70% vision tokens)
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

                st.info(f"Loaded {len(all_images)} script page(s). Evaluating...")

                # Lightweight Prompt: Gemini performs evaluation and returns simple visual intent flags
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
                            "needs_visual": "function_graph", 
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
                
                Note on "needs_visual":
                - Use "function_graph" if the question involves curves/functions.
                - Use "transformation" if the question involves rotations/reflections/grid shifts.
                - Use "workflow" if the question involves flowcharts/networks/probability trees.
                - Use "geometry" if the question involves inscribed circles/polygons.
                - Use "data_table" if the question involves tabular evaluation.
                - Use null if no special visual is needed.
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

                # Compute score metrics
                total_score = sum(item.get("score", 0) for item in student_data.get("questions", []))
                max_score = sum(item.get("max_score", 0) for item in student_data.get("questions", []))
                percentage = round((total_score / max_score) * 100, 1) if max_score > 0 else 0

                # ==========================================
                # 4. PYTHON HTML COMPOSITION ENGINE
                # ==========================================
                questions_html = ""
                for q in student_data.get("questions", []):
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
                        
                        # Injects step badge tags programmatically
                        if not is_correct and line.get("explanation"):
                            explanation_text = line["explanation"]
                            working_lines_html += f'''
                            <div class="explanation-box">
                                <span class="step-tag">[Step {step_idx}]</span> <strong>💡 Concept Guide:</strong><br/>
                                {explanation_text}
                            </div>
                            '''

                    # Map Gemini intent flags to local Python renderers
                    visual_type = q.get("needs_visual")
                    visual_html = ""

                    if visual_type == "function_graph":
                        img_b64 = generate_function_graph_b64()
                        tbl_html = generate_data_table_html()
                        visual_html = f'''
                        <div class="graph-container">
                            <img src="{img_b64}" />
                            <div style="margin-top:10px;">{tbl_html}</div>
                        </div>
                        '''
                    elif visual_type == "transformation":
                        img_b64 = generate_transformation_graph_b64(90)
                        visual_html = f'<div class="graph-container"><img src="{img_b64}" /></div>'
                    elif visual_type == "workflow":
                        img_b64 = generate_network_diagram_b64()
                        visual_html = f'<div class="graph-container"><img src="{img_b64}" /></div>'
                    elif visual_type == "geometry":
                        img_b64 = generate_geometry_patches_b64()
                        visual_html = f'<div class="graph-container"><img src="{img_b64}" /></div>'
                    elif visual_type == "data_table":
                        tbl_html = generate_data_table_html()
                        visual_html = f'<div class="graph-container">{tbl_html}</div>'

                    questions_html += f"""
                    <div class="question-block">
                        <div class="question-title">{q.get("title", "Question")} &nbsp;&nbsp;&nbsp; [{q.get("max_score", 0)} Marks]</div>
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

                # HTML Template for WeasyPrint
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
                    label="📄 Download Combined Marked PDF",
                    data=pdf_bytes,
                    file_name="combined_marked_script.pdf",
                    mime="application/pdf"
                )

            except Exception as e:
                st.error(f"Error processing scripts: {e}")
