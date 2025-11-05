import google.generativeai as genai
import json

api_key="AIzaSyDv7atwbg67P7s1ZXmQoev5x_CQVuo_ky8"
# Initialize Gemini
def init_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')




def interpret_query_gemini(model, query, schema):
    prompt = f"""
    You are a data analysis assistant. Given the query below and schema, return a JSON object describing the action.
    Schema: {schema}
    Query: {query}


    JSON format:
        {{
        "action": "eda | cleaning | outlier | feature_engineering | preprocessing",
        "columns": ["col1", "col2"],
        "details": "description of what to do"
        }}
        """
    response = model.generate_content(prompt)
    try:
        return json.loads(response.text)
    except Exception:
        return {"action": "unknown", "details": response.text}