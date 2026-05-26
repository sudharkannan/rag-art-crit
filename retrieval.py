import os
import re
import chromadb
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("art_knowledge")

def parse_frontmatter(content):
    metadata = {}
    frontmatter_match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    if frontmatter_match:
        for line in frontmatter_match.group(1).split("\n"):
            if ": " in line:
                key, value = line.split(": ", 1)
                metadata[key.strip()] = value.strip()
    return metadata

def ingest_knowledge_base(folder_path="./knowledge_base"):
    files = [f for f in os.listdir(folder_path) if f.endswith(".md")]
    
    for filename in files:
        filepath = os.path.join(folder_path, filename)
        with open(filepath, "r") as f:
            content = f.read()

        metadata = parse_frontmatter(content)
        
        clean_content = re.sub(r"^---\n.*?\n---\n", "", content, flags=re.DOTALL).strip()

        search_terms = metadata.get("search_terms", "")
        embed_text = f"{clean_content}\n\nSearch terms: {search_terms}"

        embedding = embedder.encode(embed_text).tolist()

        collection.upsert(
            documents=[clean_content],
            embeddings=[embedding],
            ids=[filename],
            metadatas=[{
                "category": metadata.get("category", ""),
                "subcategory": metadata.get("subcategory", ""),
                "level": metadata.get("level", "beginner"),
                "filename": filename
            }]
        )

    print("Database Built")

def retrieve_for_issue(category, observation, n_results=2):
    query = f"{category}: {observation}"
    query_embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"category": category}
    )

    # return list of document strings
    if results and results["documents"]:
        return results["documents"][0]
    return []

def retrieve_for_analysis(analysis):
    context = []

    for issue in analysis.get("issues", []):
        docs = retrieve_for_issue(issue["category"], issue["observation"])
        context.extend(docs)

    for suggestion in analysis.get("suggestions", []):
        docs = retrieve_for_issue(suggestion["category"], suggestion["observation"])
        context.extend(docs)

    # deduplicate while preserving order
    seen = set()
    unique_context = []
    for doc in context:
        if doc not in seen:
            seen.add(doc)
            unique_context.append(doc)

    return unique_context

if __name__ == "__main__":
    ingest_knowledge_base()