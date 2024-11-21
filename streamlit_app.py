import streamlit as st
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np

NEO4J_URI = st.secrets["NEO4J_URI"]
NEO4J_USERNAME = st.secrets["NEO4J_USERNAME"]
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]

# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Initialize SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def fetch_queries_with_objectives_and_details():
    """
    Fetch all queries along with their objectives, traffic sources, and KPIs from Neo4j.
    """
    query = """
    MATCH (q:Query)-[:MATCHES_OBJECTIVE]->(o:Objectives)
    OPTIONAL MATCH (ts:TrafficSource)-[:SUITABLE_FOR]->(o)
    OPTIONAL MATCH (q)-[:MEASURED_BY]->(k:KPI)
    RETURN 
        q.id AS query_id, 
        q.query AS query_text, 
        o.id AS objective_id, 
        o.name AS objective_name, 
        COLLECT(DISTINCT ts.name) AS traffic_sources,
        COLLECT(DISTINCT k.KPI) AS kpis
    """
    with driver.session() as session:
        result = session.run(query)
        queries = {}
        for record in result:
            query_id = record["query_id"]
            if query_id not in queries:
                queries[query_id] = {
                    "id": query_id,
                    "query": record["query_text"],
                    "objectives": [],
                }
            queries[query_id]["objectives"].append({
                "id": record["objective_id"],
                "name": record["objective_name"],
                "traffic_sources": record["traffic_sources"],
                "kpis": record["kpis"]
            })
        return list(queries.values())

def calculate_similarity(queries, input_text):
    """
    Calculate similarity between queries and input_text using SentenceTransformer.
    """
    query_texts = [q["query"] for q in queries]
    query_embeddings = model.encode(query_texts)
    input_embedding = model.encode([input_text])
    similarities = np.dot(query_embeddings, input_embedding.T).flatten()
    for i, query in enumerate(queries):
        query["similarity"] = similarities[i]
    return queries

# Streamlit App
st.title("Query Similarity and Objectives Viewer")
st.write("Enter a query to find similar queries and their related objectives, traffic sources, and KPIs.")

# Input query
input_text = st.text_input("Enter your query:", value="How can I increase online sales?")

if input_text:
    try:
        # Fetch queries and their details
        queries = fetch_queries_with_objectives_and_details()
        if not queries:
            st.write("No queries found in the database.")
        else:
            # Calculate similarity
            similar_queries = calculate_similarity(queries, input_text)
            similar_queries = sorted(similar_queries, key=lambda x: x["similarity"], reverse=True)

            # Check highest similarity score
            if similar_queries[0]["similarity"] < 0.3:
                st.warning("The input query might not be relevant to marketing.")
            else:
                # Display top results
                st.subheader("Top Similar Queries:")
                for query in similar_queries[:5]:  # Show top 5 results
                    with st.expander(f"{query['query']}"):
                        #st.write(f"**Query ID:** {query['id']} | **Similarity:** {query['similarity']:.4f}")
                        st.write("**Related Objectives:**")
                        
                        # Create a tab for each objective, using the objective name as the tab label
                        tabs = st.tabs([obj['name'] for obj in query["objectives"]])
                        for idx, obj in enumerate(query["objectives"]):
                            with tabs[idx]:
                                st.write(f"**Traffic Sources:** {', '.join(obj['traffic_sources']) if obj['traffic_sources'] else 'None'}")
                                st.write(f"**KPIs:** {', '.join(obj['kpis']) if obj['kpis'] else 'None'}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        driver.close()
