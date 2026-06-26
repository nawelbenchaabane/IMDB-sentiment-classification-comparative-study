from pathlib import Path

import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "distilbert-imdb"


@st.cache_resource
def load_model():
    """
    Load the fine-tuned DistilBERT model and tokenizer.
    The model folder must be located at:
    models/distilbert-imdb/
    """
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device


def predict_sentiment(text, tokenizer, model, device):
    """
    Predict the sentiment of a single movie review.
    Returns the predicted label, confidence score and class probabilities.
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

    st.markdown("### Model information")
    st.write("**Model:** Fine-tuned DistilBERT")
    st.write("**Dataset:** IMDB Reviews")
    st.write("**Task:** Binary sentiment classification")
    st.write(f"**Model path:** `{MODEL_PATH}`")

    if not MODEL_PATH.exists():
        st.error(
            f"Model folder not found:\n\n`{MODEL_PATH}`\n\n"
            "Please make sure the fine-tuned model is saved in:\n\n"
            "`models/distilbert-imdb/`"
        )
        st.stop()

    tokenizer, model, device = load_model()

    st.markdown("---")

    review = st.text_area(
        "Enter a movie review:",
        height=180,
        placeholder=(
            "Example: This movie was absolutely fantastic. "
            "The story was emotional and the acting was excellent."
        )
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

            st.markdown("### Prediction result")

            if label == "Positive":
                st.success(f"Prediction: {label}")
            else:
                st.error(f"Prediction: {label}")

            st.write(f"Confidence: **{confidence:.2%}**")

            st.markdown("### Class probabilities")

            st.write({
                "Negative": f"{probabilities[0]:.2%}",
                "Positive": f"{probabilities[1]:.2%}"
            })

            st.progress(float(confidence))

    st.markdown("---")
    st.caption(
        "Fine-tuned DistilBERT | IMDB Reviews Dataset | Binary Sentiment Classification"
    )


if __name__ == "__main__":
    main()
