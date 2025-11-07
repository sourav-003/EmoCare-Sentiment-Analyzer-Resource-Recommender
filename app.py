import gradio as gr
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

print("--- Loading Gradio and Transformers ---")

model_path = "your-username/mental-health-sentiment-biomedbert"

print(f"Loading trained model from Hugging Face Hub: {model_path}")
classifier = pipeline("sentiment-analysis", model=model_path)
print("Model loaded successfully from Hugging Face Hub!")

# --- Define Recommendations ---
recommendation_map = {
    "very negative": (
        "**Resource Recommendation (High Priority):**\n"
        "We understand this is an incredibly difficult time. Support is available. "
        "Please consider reaching out to a professional or a support line.\n\n"
        "* **Crisis Text Line:** Text 'HOME' to 741741\n"
        "* **988 Suicide and Crisis Lifeline:** Call 988 or visit [988lifeline.org](https://988lifeline.org/)\n"
        "* **CancerCare Support:** Call 1-800-813-4673 or visit [CancerCare.org](https://www.cancercare.org/)"
    ),
    "negative": (
        "**Resource Recommendation:**\n"
        "It's completely valid to feel worried, anxious, or frustrated. You are not alone. "
        "Connecting with others who understand can make a difference.\n\n"
        "* **Find a Support Group:** [Cancer Support Community](https://www.cancersupportcommunity.org/)\n"
        "* **For Caregivers:** [Cancer.Net - Support for Caregivers](https://www.cancer.net/coping-with-cancer/managing-emotions/support-caregivers)\n"
        "* **Managing 'Scanxiety':** [Tips from MSK Cancer Center](https://www.mskcc.org/cancer-care/patient-education/managing-scanxiety-during-your-cancer-treatment)"
    ),
    "positive": (
        "**Resource Recommendation:**\n"
        "Thank you for sharing this. Positive moments and messages of hope are so important. "
        "Your story can be a source of strength for others in this community.\n\n"
        "* **Share Your Story:** [American Cancer Society - Stories of Hope](https://www.cancer.org/treatment/survivorship-during-and-after-treatment/stories-of-hope.html)"
    ),
    "neutral": (
        "**Resource Recommendation:**\n"
        "For factual, up-to-date information, it's always best to consult trusted medical sources.\n\n"
        "* **General Cancer Information:** [National Cancer Institute](https://www.cancer.gov/)\n"
        "* **Clinical Trials:** [ClinicalTrials.gov](https://clinicaltrials.gov/)"
    )
}

# --- Prediction Function ---
def predict_sentiment_and_recommend(text):
    if not text.strip():
        return "Please enter some text.", "No recommendation."

    result = classifier(text)[0]
    sentiment_label = result['label'].lower()  # lowercase for key matching
    sentiment_score = result['score']

    recommendation_text = recommendation_map.get(sentiment_label, "No recommendation available.")

    return {sentiment_label: sentiment_score}, recommendation_text


print("Prediction function and recommendation engine are ready.")

# --- Gradio Interface Setup ---
input_text = gr.Textbox(
    lines=7,
    label="Enter a Post to Analyze",
    placeholder="e.g., 'I am feeling very scared about the upcoming scan results.'"
)

output_label = gr.Label(num_top_classes=4, label="Predicted Sentiment")
output_recommendation = gr.Markdown(label="Recommended Resources")

example_posts = [
    ["It's been 3 weeks since my surgery and I'm finally able to walk a mile. I'm feeling well and my body is healing."],
    ["My nan was just diagnosed and is starting radiation. What are some gifts I can get her that would be useful?"],
    ["I've been having blood on the toilet paper and I have no health insurance. I am 20 years old and can't stop worrying this is cancer."],
    ["My father-in-law was just diagnosed with lung cancer. We're looking for advice on what to expect."],
    ["My husband died recently from cancer that spread to his brain. I'm in shock. I feel down and alone."],
    ["It was hard enough losing my husband. Now my neighbor, who is my only support, has advanced breast cancer. The tears won't stop."],
    ["I am having surgery next month. What should I expect? The doctor said I will be in the hospital for 3-5 days."],
    ["A new experimental drug called telaprevir may help cure hepatitis C. Results were published in the New England Journal of Medicine."]
]

iface = gr.Interface(
    fn=predict_sentiment_and_recommend,
    inputs=input_text,
    outputs=[output_label, output_recommendation],
    title="💬 EmoCare: Sentiment Analyzer & Resource Recommender",
    description="This app uses a fine-tuned **Bio_ClinicalBERT** model to analyze mental health–related posts and provide resource recommendations for cancer survivors and caregivers.",
    examples=example_posts,
    theme=gr.themes.Soft()
)

print("\nLaunching Gradio interface... (This will work automatically on Hugging Face Spaces)")
iface.launch()
