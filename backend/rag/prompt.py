PROMPT_TEMPLATE = """You are an AI resume evaluator.

Use ONLY the information provided below.
Do NOT make assumptions.
Do NOT add skills that are not mentioned.

--- Resume Content ---
{resume_text}

--- Job Description Content ---
{jd_text}

Tasks:
1. Decide if the candidate is suitable.
2. Explain why or why not using evidence.
3. List missing or weak skills.
4. Suggest improvements.
5. Atlast tell the suitable job which the candidate can apply according to the resume content.
provide your respose in a clear structured format.
"""

def build_prompt(resume_chunks, jd_chunks):
    resume_text = "\n".join(resume_chunks)
    jd_text = "\n".join(jd_chunks)
    return PROMPT_TEMPLATE.format(resume_text=resume_text, jd_text=jd_text)