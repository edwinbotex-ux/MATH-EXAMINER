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
                

            except Exception as e:
                st.error(f"Error processing scripts: {e}")
