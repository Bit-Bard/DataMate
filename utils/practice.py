# test_gemini_api.py

import google.generativeai as genai

# 🔑 Replace with your Gemini API key
GEMINI_API_KEY = "AIzaSyCOBZDP4S-F3bj091NwYcW9wyb8XG8_fRM"

# Configure the API key
genai.configure(api_key=GEMINI_API_KEY)

try:
    # Initialize a Gemini model
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Send a simple test prompt
    response = model.generate_content("Hello Gemini, can you confirm you're working?")
    
    # Print model's response
    print("✅ API is working!")
    print("Response:", response.text)

except Exception as e:
    print("❌ Something went wrong:")
    print(e)
