import gradio as gr
import joblib

# =========================
# Load model & vectorizer
# =========================
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("spam_vectorizer.pkl")

# =========================
# Spam keywords
# =========================
spam_keywords = [
    "free", "win", "winner", "cash", "prize", "urgent",
    "click", "offer", "limited", "money", "credit",
    "lottery", "bonus", "guarantee"
]

# =========================
# Prediction function
# =========================
def predict_spam(message):
    if not message.strip():
        return "—", "", "Please enter an email message."

    message_vec = vectorizer.transform([message])
    pred = model.predict(message_vec)[0]
    prob = model.predict_proba(message_vec)[0][1]

    label = "SPAM" if pred == 1 else "NOT SPAM"

    found_words = [w for w in spam_keywords if w in message.lower()]

    if label == "SPAM":
        explanation = (
            f"This email is classified as SPAM because it contains "
            f"spam-related words such as: {', '.join(found_words)}."
            if found_words else
            "This email follows patterns commonly seen in spam messages."
        )
    else:
        explanation = (
            "This email does not contain common spam indicators "
            "and follows normal communication patterns."
        )

    return label, f"{prob:.2f}", explanation


# =========================
# CSS Styling (HF SAFE)
# =========================
css = """
body {
    background-color: #fde2ea;
}
/* Email input */
#email_box textarea {
    background-color: #ffe4ef;
    color: #7a0c3a !important;
    border-radius: 14px;
    padding: 14px;
    font-size: 15px;
}
#email_box textarea::placeholder {
    color: #9c2f5d;
}
/* Prediction box */
#prediction_box {
    background: linear-gradient(135deg, #c2185b, #ff4d6d);
    color: white;
    border-radius: 18px;
    padding: 22px;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
}
/* Probability box */
#prob_box textarea {
    background-color: #ffd1e1;
    color: #7a0c3a !important;
    border-radius: 14px;
    font-weight: bold;
    text-align: center;
    font-size: 16px;
}
/* Explanation box */
#explain_box textarea {
    background-color: #ffffff;
    color: #212529 !important;
    border-radius: 14px;
    font-size: 14px;
    padding: 12px;
}
/* Button */
#check_btn {
    background: linear-gradient(90deg, #c2185b, #ff4d6d);
    color: white;
    font-size: 16px;
    font-weight: bold;
    border-radius: 18px;
    padding: 14px;
    border: none;
}
"""


# =========================
# Gradio UI
# =========================
with gr.Blocks(css=css) as app:

    # 🔥 Title (HTML = no override)
    gr.HTML("""
        <div style="
            text-align:center;
            font-size:36px;
            font-weight:800;
            letter-spacing:1px;
            color:#c2185b;
            margin-bottom:20px;">
            Spam Email Detector
        </div>
    """)

    message_input = gr.Textbox(
        lines=5,
        label="Email / Message",
        placeholder="Type or paste your email here...",
        elem_id="email_box"
    )

    with gr.Row():
        pred_output = gr.Label(
            label="Prediction",
            elem_id="prediction_box"
        )
        prob_output = gr.Textbox(
            label="Spam Probability",
            placeholder="0.00",
            elem_id="prob_box"
        )

    explanation_output = gr.Textbox(
        label="Why this result?",
        lines=3,
        elem_id="explain_box"
    )

    submit_btn = gr.Button(
        "Check Spam",
        elem_id="check_btn"
    )

    submit_btn.click(
        fn=predict_spam,
        inputs=message_input,
        outputs=[pred_output, prob_output, explanation_output]
    )

app.launch()
