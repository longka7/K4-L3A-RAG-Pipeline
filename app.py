import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import generate_with_citation


load_dotenv()

st.set_page_config(
    page_title="VinUni Scholarship Assistant",
    page_icon="🎓",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("VinUni Scholarship Assistant")
    st.caption("Hỏi đáp dựa trên tài liệu học bổng và hỗ trợ tài chính.")
    top_k = st.slider("Số chunks", 3, 10, 5)

st.title("Học bổng & hỗ trợ tài chính")
st.caption("Câu trả lời được tạo từ corpus nội bộ và luôn hiển thị nguồn đối chiếu.")


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Nguồn tham khảo ({len(sources)})"):
        for index, source in enumerate(sources, 1):
            metadata = source["metadata"]
            title = metadata["title"]
            url = metadata.get("url")
            label = f"[Document {index}] {title} — {metadata['source']}"
            if url:
                st.markdown(f"{label}  \\n[{url}]({url})")
            else:
                st.markdown(label)
            st.caption(f"Score: {float(source['score']):.4f} | Chunk: {metadata['chunk_index']}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

query = st.chat_input("Nhập câu hỏi...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        result = generate_with_citation(query, top_k=top_k)
        answer = result["answer"]
        sources = result["sources"]
        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
