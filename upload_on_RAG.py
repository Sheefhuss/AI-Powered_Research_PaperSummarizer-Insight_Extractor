import json
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

documents = []
metadatas = []

parsed_folder = "Final_parsed_output"

for filename in os.listdir(parsed_folder):
    if filename.endswith(".json"):
        file_path = os.path.join(parsed_folder, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            metadata = data.get("metadata", {})
            insight = data.get("insigth") or {}  

            text = f"""
            Title: {metadata.get('title', '')}
            Authors: {", ".join(metadata.get('authors', []))}
            Publication Year: {metadata.get('publication_year', '')}
            DOI: {metadata.get('doi', '')}
            Keywords: {", ".join(metadata.get('keywords', []))}
            
            Domain: {", ".join(insight.get("domain", []))}
            Research Problem: {insight.get("research_problem", "")}
            Methods: {", ".join(insight.get("methods", []))}
            Datasets: {", ".join(insight.get("datasets", []))}
            Metrics: {", ".join(insight.get("metrics", []))}
            Key Findings: {insight.get("key_findings", "")}
            Limitations: {insight.get("limitations", "")}
            Future Directions: {insight.get("future_directions", "")}
            
            Abstract:
            {data.get("abstract", "")}
            
            Summary:
            {data.get("summary", "")}
            """

            documents.append(text)

            metadatas.append({
                "paper_id": data.get("document_id"),
                "title": metadata.get("title"),
                "source": data.get("source_file"),
                "publication_year": metadata.get("publication_year"),
                "domain": ", ".join(insight.get("domain", [])) if insight.get("domain") else "Unknown",
            })

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

try:
    with open("arxiv_papers_vb.json", "r", encoding="utf-8") as f:
        arxiv_papers = json.load(f)
        
    for paper in arxiv_papers:
        insight = paper.get("insight") or {}
        
        text = f"""
        Title: {paper.get('title', '')}
        Authors: {paper.get('authors', '')}
        Published: {paper.get('published', '')}
        Categories: {paper.get('categories', '')}
        
        Domain: {", ".join(insight.get("domain", []))}
        Research Problem: {insight.get("research_problem", "")}
        Methods: {", ".join(insight.get("methods", []))}
        Datasets: {", ".join(insight.get("datasets", []))}
        Metrics: {", ".join(insight.get("metrics", []))}
        Key Findings: {insight.get("key_findings", "")}
        Limitations: {insight.get("limitations", "")}
        Future Directions: {insight.get("future_directions", "")}
        
        Abstract:
        {paper.get("abstract", "")}
        """
        
        documents.append(text)
        
        metadatas.append({
            "paper_id": paper.get("paper_id"),
            "title": paper.get("title"),
            "source": paper.get("source"),
            "categories": paper.get("categories"),
            "domain": ", ".join(insight.get("domain", [])) if insight.get("domain") else "Unknown",
        })
except FileNotFoundError:
    pass

try:
    with open("pubmed_multiple_queries1.json", "r", encoding="utf-8") as f:
        pubmed_papers = json.load(f)
        
    for paper in pubmed_papers:
        insight = paper.get("insight") or {}
        raw_keywords = paper.get('keywords', [])
        formatted_keywords = ", ".join(raw_keywords) if isinstance(raw_keywords, list) else raw_keywords
        
        text = f"""
        Title: {paper.get('title', '')}
        Authors: {paper.get('authors', '')}
        Journal: {paper.get('journal', '')}
        Keywords: {formatted_keywords}
        
        Domain: {", ".join(insight.get("domain", []))} 
        Research Problem: {insight.get("research_problem", "")}
        Methods: {", ".join(insight.get("methods", []))}
        Datasets: {", ".join(insight.get("datasets", []))}
        Metrics: {", ".join(insight.get("metrics", []))}
        Key Findings: {insight.get("key_findings", "")}
        Limitations: {insight.get("limitations", "")}
        Future Directions: {insight.get("future_directions", "")}
        
        Abstract:
        {paper.get("abstract", "")}
        """
        
        documents.append(text)
        
        metadatas.append({
            "paper_id": paper.get("pmid"),
            "title": paper.get("title"),
            "source": "pubmed",
            "journal": paper.get('journal', ''),
            "domain": ", ".join(insight.get("domain", [])) if insight.get("domain") else "Unknown",
        })
except FileNotFoundError:
    pass
    
if documents:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_texts(texts=documents, embedding=embeddings, metadatas=metadatas)
    print(f"Number of vectors in index: {vector_db.index.ntotal}")
    vector_db.save_local("research_papers_faiss")