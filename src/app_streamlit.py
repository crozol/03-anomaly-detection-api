"""Demo interactiva con Streamlit: subir CSV y visualizar anomalías."""

from __future__ import annotations


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Anomaly Detection", layout="wide")
    st.title("Anomaly Detection — NASA Turbofan")
    st.caption("Sube un CSV con una serie multivariada y el autoencoder marcará las ventanas anómalas.")

    uploaded = st.file_uploader("CSV de entrada", type=["csv"])
    if uploaded is None:
        st.info("Esperando archivo…")
        return

    # TODO: leer CSV, pasar por el modelo, graficar serie original + score + ventanas anómalas
    st.warning("Pipeline de inferencia aún no implementado.")


if __name__ == "__main__":
    main()
