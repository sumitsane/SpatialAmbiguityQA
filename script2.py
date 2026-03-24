from neo4j import GraphDatabase
import ollama
import time
import csv
import os

# Move heavy imports & model loading OUTSIDE the loop/function
try:
    from sentence_transformers import SentenceTransformer, util
    embedder = SentenceTransformer('all-MiniLM-L6-v2')  # load once
    print("SentenceTransformer model loaded successfully")
except ImportError as e:
    print("Cannot load sentence-transformers:", e)
    embedder = None  # fallback or exit

# Neo4j connection (consider using env vars in production)
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "Nadmin123")
)

def load_queries():
    """Load ambiguous queries from Neo4j"""
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
                    "meaning": r["meaning"] or ""  # handle null
                })
    except Exception as e:
        print(f"Neo4j error: {e}")
    return data

def load_control_queries(csv_path="1089Control.csv"):
    """Load 25 control queries from CSV (no header expected)"""
    data = []
    if not os.path.exists(csv_path):
        print(f"Control CSV not found: {csv_path}")
        return data

    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:  # utf-8-sig removes BOM
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    query, ambiguity, meaning = row[0].strip(), row[1].strip(), row[2].strip()
                    data.append({
                        "query": query,
                        "ambiguity": ambiguity or "Control (GeoQuestions1089)",
                        "meaning": meaning or "Expected correct answer"
                    })
                elif len(row) == 1 and row[0].strip():  # skip empty lines
                    continue
    except Exception as e:
        print(f"Error reading control CSV: {e}")
    print(f"Loaded {len(data)} control queries from {csv_path}")
    return data

def machine_only(query):
    try:
        response = ollama.chat(
            model="mistral:7b",
            messages=[{"role": "user", "content": query}]
        )
        return response["message"]["content"]
    except Exception as e:
        return f"[Ollama error: {str(e)}]"

def evaluate(control=False):
    if embedder is None:
        print("Cannot run evaluation: sentence-transformers not available")
        return

    if control:
        dataset = load_control_queries()
        results_file = "control_machine_only_results.csv"
        acc_file = "control_ambiguity_type_accuracy.csv"
        print("=== Running CONTROL set (GeoQuestions1089 sample) ===")
    else:
        dataset = load_queries()
        results_file = "machine_only_results.csv"
        acc_file = "ambiguity_type_accuracy.csv"
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
        prediction = machine_only(query)
        end = time.time()
        elapsed = end - start
        times.append(elapsed)

        # semantic similarity
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
            query,
            ambiguity,
            gt,
            prediction[:500] + "..." if len(prediction) > 500 else prediction,
            status,
            elapsed,
            round(sim, 4)
        ])

        print(f"Done: {query} | sim={sim:.3f} | {status}")

    # Final metrics
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

    # Save results
    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Query", "Ambiguity Type", "Ground Truth", "Prediction", "Status", "Time", "Similarity"])
        writer.writerows(results_table)

    with open(acc_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Ambiguity Type", "Total", "Correct", "Accuracy"])
        writer.writerows(type_accuracy_rows)

    print(f"\nCSV files saved: {results_file} and {acc_file}")

# Run both modes
if __name__ == "__main__":
    print("Running LOCAL dataset...")
    evaluate(control=False)
    print("\n" + "="*60 + "\n")
    print("Running CONTROL dataset...")
    evaluate(control=True)

    driver.close()