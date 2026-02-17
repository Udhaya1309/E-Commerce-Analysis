"""
Demo: Run the AI Marketing Agent on 3 sample customers (Loyal, At-risk, New).
Shows Prediction, Strategy, and Generated Email for each.
"""

from marketing_agent import MarketingAIAgent, get_sample_customers


def main():
    print("=" * 60)
    print("AI Marketing Agent - Demo")
    print("=" * 60)

    agent = MarketingAIAgent()
    print("\n1. Preprocessing data...")
    agent.preprocess_data()
    print("\n2. Training model...")
    agent.train_model()

    samples = get_sample_customers()
    labels = ["Loyal customer (Sarah)", "At-risk customer (James)", "New customer (Alex)"]

    for i, (customer, label) in enumerate(zip(samples, labels), 1):
        print("\n" + "=" * 60)
        print(f"Sample {i}: {label}")
        print("=" * 60)
        result = agent.run_agent(customer)
        print(f"\nPrediction: {result['prediction']}")
        print(f"Confidence: {result['probability']:.2%}")
        print(f"Strategy:   {result['strategy']}")
        print("\n--- Generated Email ---")
        print(result["email"])
        print("--- End Email ---\n")

    print("=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
