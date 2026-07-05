"""Streamlit demo — enter a product description + image URL, get a price.

Run with: streamlit run frontend/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.inference.predictor import Predictor
from src.utils.config import load_config
from src.utils.exceptions import CheckpointError, ConfigError, InferenceError
from src.utils.logging import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="Price Predictor", page_icon="💲")
st.title("Multimodal Product Price Predictor")
st.caption("Predicts price from product text + image alone — no brand lookups, no price database.")


@st.cache_resource
def load_predictor():
    config = load_config("configs/base.yaml")
    checkpoint_path = f"{config['checkpoint_dir']}/best.pt"
    return Predictor(config, checkpoint_path)


text = st.text_area("Product description", placeholder="Wireless noise-canceling headphones, 40-hour battery, Bluetooth 5.3")
image_url = st.text_input("Image URL")

if st.button("Predict price", type="primary"):
    if not text.strip() or not image_url.strip():
        st.warning("Please provide both a description and an image URL.")
    else:
        try:
            with st.spinner("Loading model..."):
                predictor = load_predictor()
            with st.spinner("Predicting..."):
                price = predictor.predict_one(text, image_url)
            st.image(image_url, width=250)
            st.metric("Predicted price", f"${price:.2f}")
        except (ConfigError, CheckpointError) as e:
            st.error(f"Model could not be loaded: {e}")
            logger.error("Model load failure: %s", e)
        except InferenceError as e:
            st.error(f"Prediction failed: {e}")
            logger.error("Inference failure: %s", e)
