import streamlit as st
import chromadb
import os

st.set_page_config(
    page_title="Semantic Search",
    page_icon="🔍",
    layout="wide"
)


# --------------------------------------------------
# ChromaDB Setup
# --------------------------------------------------

@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path="./search_db")
    return client.get_or_create_collection(name="course_docs")


collection = get_collection()


# --------------------------------------------------
# Document Loading & Chunking
# --------------------------------------------------

def load_and_chunk(directory, chunk_size=400, overlap=50):
    """Load text files and split them into chunks."""

    chunks = []

    for filename in sorted(os.listdir(directory)):

        if not filename.endswith((".txt", ".md")):
            continue

        filepath = os.path.join(directory, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Split by paragraphs
                # Split text into chunks with overlap
        step = chunk_size - overlap

        for start in range(0, len(content), step):
            chunk_text = content[start:start + chunk_size].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "source": filename,
                    "chunk_id": f"{filename}_{start}",
                    "chunk_index": start // step,
                })

    return chunks


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.title("📁 Document Manager")

    # ----------------------------------------------
    # Re-index Documents
    # ----------------------------------------------

    if st.button("🔄 Re-index Documents"):

        chunks = load_and_chunk("docs")

        if chunks:

            # Remove existing documents so renamed files
            # don't leave old documents in the database.
            existing_ids = collection.get()["ids"]

            if existing_ids:
                collection.delete(ids=existing_ids)

            # Add the new documents
            collection.upsert(
                documents=[c["text"] for c in chunks],
                metadatas=[
                    {
                        "source": c["source"],
                        "chunk_index": str(c["chunk_index"])
                    }
                    for c in chunks
                ],
                ids=[c["chunk_id"] for c in chunks],
            )

            st.success(
                f"Indexed {len(chunks)} chunks from "
                f"{len(set(c['source'] for c in chunks))} files"
            )

        else:
            st.warning(
                "No .txt or .md files found in docs/ folder"
            )

    # ----------------------------------------------
    # Document Count
    # ----------------------------------------------

    st.metric(
        "Documents in DB",
        collection.count()
    )

    st.divider()

    # ----------------------------------------------
    # Source Filter
    # ----------------------------------------------

    # Get all sources currently in ChromaDB
    all_data = collection.get()

    source_options = sorted(
        set(
            metadata["source"]
            for metadata in all_data["metadatas"]
        )
    )

    selected_sources = st.multiselect(
        "Filter by source",
        options=source_options,
        help="Choose which files to search."
    )

    st.divider()

    # ----------------------------------------------
    # Number of Results
    # ----------------------------------------------

    n_results = st.slider(
        "Results to show",
        1,
        10,
        5
    )


# --------------------------------------------------
# Main Search Interface
# --------------------------------------------------

st.title("🔍 Semantic Search")

st.write(
    "Search your course documents by meaning, "
    "not just keywords."
)

query = st.text_input(
    "Enter your search query",
    placeholder="How does authentication work?"
)


# --------------------------------------------------
# Search
# --------------------------------------------------

if query and collection.count() > 0:

    # ----------------------------------------------
    # Build ChromaDB filter
    # ----------------------------------------------

    search_kwargs = {
        "query_texts": [query],
        "n_results": min(n_results, collection.count())
    }

    # Only add "where" when the user selected sources.
    if selected_sources:

        search_kwargs["where"] = {
            "source": {
                "$in": selected_sources
            }
        }

    # ----------------------------------------------
    # Run search
    # ----------------------------------------------

    results = collection.query(**search_kwargs)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # ----------------------------------------------
    # Result count
    # ----------------------------------------------

    st.write(
    f"Showing {len(documents)} of "
    f"{collection.count()} total chunks"
)

    st.subheader(
        f"Top {len(documents)} Results"
    )

    # ----------------------------------------------
    # Display results
    # ----------------------------------------------

    for i in range(len(documents)):

        doc = documents[i]
        metadata = metadatas[i]
        distance = distances[i]

        # Color-code relevance
        if distance < 0.5:
            relevance = "🟢 High"
            badge_color = "#28a745"


        elif distance < 1.0:
            relevance = "🟡 Medium"
            badge_color = "#ffc107"

        else:
            relevance = "🔴 Low"
            badge_color = "#dc3545"

        with st.container():

            col_meta, col_score = st.columns([3, 1])

            with col_meta:
                st.write(
                    f"**{metadata['source']}** "
                    f"— chunk {metadata['chunk_index']}"
                )

            with col_score:
                st.markdown(
                    f"""
                    <span style="
                        background-color: {badge_color};
                        color: white;
                        padding: 4px 10px;
                        border-radius: 12px;
                        font-size: 14px;
                        font-weight: bold;
                    ">
                        {relevance}
                    </span>
                    <br>
                    <small>Distance: {distance:.3f}</small>
                    """,
                    unsafe_allow_html=True
                )

            # --------------------------------------
            # Preview
            # --------------------------------------

            preview = doc[:150]

            if len(doc) > 150:
                preview += "..."

            st.write(preview)

            # --------------------------------------
            # Expandable full text
            # --------------------------------------

            with st.expander("📖 View full text"):
                st.write(doc)

            st.divider()


elif collection.count() == 0:

    st.info(
        "👈 Click 'Re-index Documents' in the sidebar "
        "to load your documents first."
    )