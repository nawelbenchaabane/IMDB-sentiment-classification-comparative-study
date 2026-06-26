import os

import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_PATH = "models/distilbert-imdb"


@st.cache_resource
def load_model():
    """
    Load the fine-tuned DistilBERT model and tokenizer.
    The model must be saved locally in models/distilbert-imdb.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device


def predict_sentiment(text, tokenizer, model, device):
    """
    Predict sentiment for a single movie review.
    Returns the predicted label and confidence score.
    """
    encoded_input = tokenizer(
        text,
        truncation=True,
        max_length=256,
        padding=True,
        return_tensors="pt"
    )

    encoded_input = {
        key: value.to(device)
        for key, value in encoded_input.items()
        if key in ["input_ids", "attention_mask"]
    }

    with torch.no_grad():
        outputs = model(**encoded_input)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)

    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()

    label_map = {
        0: "Negative",
        1: "Positive"
    }

    return label_map[predicted_class], confidence, probabilities[0].cpu().numpy()


def main():
    st.set_page_config(
        page_title="IMDB Sentiment Classifier",
        page_icon="🎬",
        layout="centered"
    )

    st.title("🎬 IMDB Sentiment Classification")
    st.write(
        "This application predicts whether a movie review is **positive** or **negative** "
        "using a fine-tuned DistilBERT model."
    )

    if not os.path.exists(MODEL_PATH):
        st.error(
            "Model folder not found. Please make sure the fine-tuned model is saved in "
            "`models/distilbert-imdb/`."
        )
        st.stop()

    tokenizer, model, device = load_model()

    review = st.text_area(
        "Enter a movie review:",
        height=180,
        placeholder="Example: This movie was absolutely fantastic. The story was emotional and the acting was excellent."
    )

    if st.button("Predict sentiment"):
        if review.strip() == "":
            st.warning("Please enter a review before prediction.")
        else:
            label, confidence, probabilities = predict_sentiment(
                review,
                tokenizer,
                model,
                device
            )

            if label == "Positive":
                st.success(f"Prediction: {label}")
            else:
                st.error(f"Prediction: {label}")

            st.write(f"Confidence: **{confidence:.2%}**")

            st.subheader("Class probabilities")
            st.write({
                "Negative": f"{probabilities[0]:.2%}",
                "Positive": f"{probabilities[1]:.2%}"
            })

    st.markdown("---")
    st.caption(
        "Model: Fine-tuned DistilBERT | Dataset: IMDB Reviews | Task: Binary Sentiment Classification"
    )


if __name__ == "__main__":
    main()