from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from inference import predict, prepare_image_for_model
from model import load_model


MODEL_PATH = "models/quickdraw_model.pt"


def main():
    st.set_page_config(page_title="Scribble AI")
    st.title("Scribble AI")
    st.caption("Draw a doodle and let a small PyTorch model classify it.")

    with st.sidebar:
        st.subheader("Canvas")
        stroke_width = st.slider("Stroke width", 1, 25, 8)
        stroke_color = st.color_picker("Stroke color", "#000000")
        bg_color = st.color_picker("Background color", "#ffffff")
        realtime_update = st.checkbox("Update in realtime", True)
        st.caption(f"Model: `{MODEL_PATH}`")

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        update_streamlit=realtime_update,
        height=500,
        width=500,
        drawing_mode="freedraw",
        point_display_radius=0,
        key="canvas",
    )

    if canvas_result.image_data is None:
        return

    if st.button("Predict", type="primary"):
        image = Image.fromarray(canvas_result.image_data.astype("uint8"))
        resized_image, model_input = prepare_image_for_model(image)

        preview_col, prediction_col = st.columns([1, 2])
        with preview_col:
            st.image(resized_image, caption="28x28 model input")

        try:
            model, class_names = load_model(MODEL_PATH)
        except FileNotFoundError:
            st.error(
                "No trained model found. Run "
                "`uv run python train.py --data-dir data/quickdraw` first."
            )
            return

        predicted_class, confidence = predict(model, class_names, model_input)
        with prediction_col:
            st.metric("Prediction", predicted_class, f"{confidence:.1%} confidence")
            st.caption(f"Classes: {', '.join(class_names)}")


if __name__ == "__main__":
    main()
