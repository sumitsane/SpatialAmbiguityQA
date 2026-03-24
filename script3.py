from neo4j import GraphDatabase
import ollama
import time
import csv
import os

# ─────────────────────────────────────────
# Heavy imports & model loading (once)
# ─────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer, util
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("SentenceTransformer model loaded successfully")
except ImportError as e:
    print("Cannot load sentence-transformers:", e)
    embedder = None

# Neo4j connection
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "Nadmin123")
)

# ─────────────────────────────────────────
# Load queries
# ─────────────────────────────────────────
def load_queries():
    cypher = """
    MATCH (q:GeoQuery)-[:HAS_AMBIGUITY]->(a:AmbiguityType)
    MATCH (q)-[:INTENDED_MEANING]->(i:IntendedInterpretation)
    RETURN q.text AS query, a.name AS ambiguity, i.meaning AS meaning
    """
    data = []
    try:
        with driver.session() as session:
            results = session.run(cypher)
            for r in results:
                data.append({
                    "query": r["query"],
                    "ambiguity": r["ambiguity"],
                    "meaning": r["meaning"] or ""
                })
    except Exception as e:
        print(f"Neo4j error: {e}")
    return data

def load_control_queries(csv_path="1089Control.csv"):
    data = []
    if not os.path.exists(csv_path):
        print(f"Control CSV not found: {csv_path}")
        return data

    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    query, ambiguity, meaning = row[0].strip(), row[1].strip(), row[2].strip()
                    data.append({
                        "query": query,
                        "ambiguity": ambiguity or "Control (GeoQuestions1089)",
                        "meaning": meaning or "Expected correct answer"
                    })
    except Exception as e:
        print(f"Error reading control CSV: {e}")

    print(f"Loaded {len(data)} control queries from {csv_path}")
    return data

# ─────────────────────────────────────────
# 🚫 LEAKAGE-FREE RETRIEVAL (NO SPLIT)
# ─────────────────────────────────────────
def retrieve_relevant_examples(query, all_examples, top_k=5):
    """
    Prevents leakage WITHOUT dataset splitting by removing:
    - exact query
    - near-duplicate queries
    - examples whose meaning is too semantically aligned
    """

    if embedder is None or not all_examples:
        return ""

    query_clean = query.strip().lower()
    query_emb = embedder.encode(query, convert_to_tensor=True)

    filtered = []

    for ex in all_examples:
        ex_query = ex["query"].strip().lower()
        ex_meaning = (ex["meaning"] or "").strip()

        # ❌ 1. Remove exact query
        if ex_query == query_clean:
            continue

        # ❌ 2. Remove near-duplicate queries
        ex_query_emb = embedder.encode(ex_query, convert_to_tensor=True)
        sim_query = util.cos_sim(query_emb, ex_query_emb)[0][0].item()

        if sim_query > 0.85:
            continue

        # ❌ 3. Remove meaning leakage (key fix)
        if ex_meaning:
            ex_meaning_emb = embedder.encode(ex_meaning, convert_to_tensor=True)
            sim_meaning = util.cos_sim(query_emb, ex_meaning_emb)[0][0].item()

            if sim_meaning > 0.75:
                continue

        filtered.append(ex)

    if not filtered:
        return ""

    # Build embeddings AFTER filtering
    texts = [f"Q: {ex['query']} → Meaning: {ex['meaning']}" for ex in filtered]
    embs = embedder.encode(texts, convert_to_tensor=True)

    sims = util.cos_sim(query_emb, embs)[0]
    top_idx = sims.topk(min(top_k, len(texts))).indices.tolist()

    selected = [texts[i] for i in top_idx]
    return "\n".join(selected)

# ─────────────────────────────────────────
# Hybrid model
# ─────────────────────────────────────────
def hybrid_model(query, all_examples):
    examples_context = retrieve_relevant_examples(query, all_examples, top_k=5)

    prompt = f"""You are a strict geospatial query disambiguation expert for Jharkhand, India.

You are given the most relevant annotated examples from the knowledge graph.
Each example shows: Q: [query] → Meaning: [exact intended meaning]

For the new User Query, output ONLY the exact intended meaning as ONE short sentence.

Rules:
- Return ONLY the meaning sentence
- No explanations
- Do not repeat examples

Examples:
{examples_context}

User Query: {query}

Exact intended meaning:"""

    try:
        response = ollama.chat(
            model="mistral:7b",
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.0,
                "top_p": 0.9,
                "num_ctx": 8192,
                "stop": ["\n\n", "User Query:", "Examples:"]
            }
        )

        raw = response["message"]["content"].strip()

        # Cleanup prefixes
        for prefix in ["Correct interpretation:", "Answer:", "Meaning:", "Exact intended meaning:"]:
            if raw.lower().startswith(prefix.lower()):
                raw = raw[len(prefix):].strip()

        if '\n' in raw:
            raw = raw.split('\n')[0].strip()

        return raw

    except Exception as e:
        return f"[Ollama error: {str(e)}]"

# ─────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────
def evaluate(control=False):
    if embedder is None:
        print("Cannot run evaluation: sentence-transformers not available")
        return

    all_examples = load_queries()
    print(f"Loaded {len(all_examples)} examples from Neo4j")

    if control:
        dataset = load_control_queries()
        results_file = "control_hybrid_rag_results.csv"
        acc_file = "control_hybrid_ambiguity_type_accuracy.csv"
        print("=== Running CONTROL set ===")
    else:
        dataset = all_examples
        results_file = "hybrid_rag_results.csv"
        acc_file = "hybrid_ambiguity_type_accuracy.csv"
        print("=== Running LOCAL ambiguous queries set ===")

    if not dataset:
        print("No queries loaded – aborting.")
        return

    total = len(dataset)
    correct = 0
    hallucination = 0
    type_correct = {}
    type_total = {}
    times = []
    results_table = []

    for item in dataset:
        query = item["query"]
        gt = item["meaning"]
        ambiguity = item["ambiguity"]

        type_total[ambiguity] = type_total.get(ambiguity, 0) + 1

        start = time.time()
        prediction = hybrid_model(query, all_examples)
        elapsed = time.time() - start
        times.append(elapsed)

        sim = util.cos_sim(
            embedder.encode(gt, convert_to_tensor=True),
            embedder.encode(prediction, convert_to_tensor=True)
        )[0][0].item()

        if sim > 0.65:
            correct += 1
            type_correct[ambiguity] = type_correct.get(ambiguity, 0) + 1
            status = "correct"
        else:
            hallucination += 1
            status = "hallucinated"

        results_table.append([
            query, ambiguity, gt,
            prediction[:500] + "..." if len(prediction) > 500 else prediction,
            status, elapsed, round(sim, 4)
        ])

        print(f"Done: {query[:60]:<60} | sim={sim:.3f} | {status}")

    accuracy = correct / total if total > 0 else 0
    hallucination_rate = hallucination / total if total > 0 else 0
    avg_time = sum(times) / len(times) if times else 0

    print("\n----------------------------")
    print(f"TOTAL QUERIES: {total}")
    print(f"ACCURACY: {accuracy:.4f}")
    print(f"HALLUCINATION RATE: {hallucination_rate:.4f}")
    print(f"AVERAGE TIME PER QUERY: {avg_time:.2f}s")
    print("----------------------------")

    print("\nAccuracy by Ambiguity Type:\n")
    type_accuracy_rows = []

    for t in sorted(type_total.keys()):
        cnt = type_total[t]
        c = type_correct.get(t, 0)
        acc = c / cnt if cnt > 0 else 0
        print(f"{t}: {acc:.4f} ({c}/{cnt})")
        type_accuracy_rows.append([t, cnt, c, acc])

    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Query", "Ambiguity Type", "Ground Truth", "Prediction", "Status", "Time", "Similarity"])
        writer.writerows(results_table)

    with open(acc_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Ambiguity Type", "Total", "Correct", "Accuracy"])
        writer.writerows(type_accuracy_rows)

    print(f"\nCSV files saved: {results_file} and {acc_file}")

# ─────────────────────────────────────────
# Run
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("Running LOCAL dataset (LEAKAGE-CONTROLLED)...")
    evaluate(control=False)

    print("\n" + "="*80 + "\n")

    print("Running CONTROL dataset...")
    evaluate(control=True)

    driver.close()