import google.generativeai as genai
genai.configure(api_key="AIzaSyBBv9qZyzwaXRSR1m0yPkpvYZSKFhdNP80")


# Use the latest available fast model
model = genai.GenerativeModel("models/gemini-2.5-flash")

# Simple test prompt
prompt = "Give feedback on how to improve squats based on body pose keypoints."

response = model.generate_content(prompt)
print(response.text)