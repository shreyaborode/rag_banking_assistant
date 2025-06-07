from rag_chain.rag_model import build_rag_pipeline

def main():
    print("🔍 Banking RAG Assistant Started...")
    qa = build_rag_pipeline()

    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query.lower() == "exit":
            break
        answer = qa.run(query)
        print("\n💡 Answer:\n", answer)

if __name__ == "__main__":
    main()
