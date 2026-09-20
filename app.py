import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import generate_with_citation


load_dotenv()

st.set_page_config(
    page_title="VinUni Scholarship Chatbot",
    page_icon="",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("VinUni Scholarship Chatbot")
    st.caption("Hỏi đáp về học bổng & hỗ trợ tài chính VinUniversity")
    top_k = st.slider("Số chunks", 3, 10, 5)

st.title("VinUni Scholarship Chatbot")
st.caption(
    "Trả lời dựa trên tài liệu chính sách và tin tức học bổng VinUniversity đã thu thập. "
    "Mọi câu trả lời đều kèm nguồn trích dẫn."
)


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Nguồn tham khảo ({len(sources)})"):
        for index, source in enumerate(sources, 1):
            metadata = source.get("metadata", {})
            title = metadata.get("title") or metadata.get("source") or source.get("id")
            file_source = metadata.get("source", "unknown")
            method = source.get("retrieval_method", "unknown")
            score = source.get("score", 0.0)
            st.markdown(
                f"**{index}. {title}** — `{file_source}` "
                f"(method: `{method}`, score: `{score:.3f}`)"
            )
            preview = (source.get("content") or "")[:300].replace("\n", " ")
            st.caption(preview + ("..." if len(source.get("content") or "") > 300 else ""))


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

query = st.chat_input("Nhập câu hỏi về học bổng VinUni...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm và tạo câu trả lời..."):
            result = generate_with_citation(query, top_k=top_k)
        answer = result["answer"]
        sources = result["sources"]
        st.markdown(answer)
        st.caption(f"retrieval_source: `{result['retrieval_source']}`")
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
