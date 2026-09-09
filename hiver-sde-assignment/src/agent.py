import pandas as pd
import joblib
import re


# ============================================================
# FILES
# ============================================================

MODEL_DIR = "models"

CLUSTER_FILE = "data/intent_clusters.csv"


# ============================================================
# LOAD SEARCH MODEL
# ============================================================

print("Loading AI search model...")

vectorizer = joblib.load(
    f"{MODEL_DIR}/vectorizer.pkl"
)

search_model = joblib.load(
    f"{MODEL_DIR}/search_model.pkl"
)

messages = pd.read_pickle(
    f"{MODEL_DIR}/messages.pkl"
)

print("Search model loaded successfully!")


# ============================================================
# LOAD INTENT CLUSTERS
# ============================================================

clusters = pd.read_csv(CLUSTER_FILE)

clusters = clusters.dropna(
    subset=["text", "cluster"]
).copy()

clusters["text"] = clusters["text"].astype(str)

clusters["tweet_id"] = (
    clusters["tweet_id"]
    .astype(str)
)


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+",
        " ",
        text
    )

    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# INTENT NAMES
# ============================================================

INTENT_NAMES = {

    0: "Order / Delivery Issue",

    1: "Refund / Return Issue",

    2: "Payment / Billing Issue",

    3: "Account Issue",

    4: "Product Issue",

    5: "Shipping / Tracking Issue",

    6: "Cancellation Request",

    7: "Technical / App Issue",

    8: "General Customer Support",

    9: "Other"
}


# ============================================================
# SEARCH CUSTOMER MESSAGES
# ============================================================

def search_customer_messages(
    query,
    top_k=5
):

    query = clean_text(query)

    query_vector = vectorizer.transform(
        [query]
    )

    distances, indices = search_model.kneighbors(
        query_vector,
        n_neighbors=top_k
    )

    results = []

    for distance, index in zip(
        distances[0],
        indices[0]
    ):

        row = messages.iloc[index]

        result = {

            "tweet_id": str(
                row["tweet_id"]
            ),

            "author_id": row["author_id"],

            "text": row["text"],

            "similarity": round(
                1 - distance,
                4
            )
        }

        # Keep response_tweet_id if it exists
        if "response_tweet_id" in row.index:

            result["response_tweet_id"] = str(
                row["response_tweet_id"]
            )

        else:

            result["response_tweet_id"] = ""

        results.append(result)

    return results


# ============================================================
# FIND INTENT FROM CLUSTER DATA
# ============================================================

def find_intent(tweet_id):

    tweet_id = str(tweet_id)

    match = clusters[
        clusters["tweet_id"] == tweet_id
    ]

    if len(match) > 0:

        cluster_id = int(
            match.iloc[0]["cluster"]
        )

        return (
            INTENT_NAMES.get(
                cluster_id,
                "General Customer Support"
            ),
            cluster_id
        )


    return (
        "General Customer Support",
        -1
    )


# ============================================================
# CREATE TWEET LOOKUP
# ============================================================

tweet_lookup = {}

for _, row in messages.iterrows():

    tweet_id = str(
        row["tweet_id"]
    )

    tweet_lookup[tweet_id] = row["text"]


# ============================================================
# GET HISTORICAL AMAZON RESPONSE
# ============================================================

def get_historical_response(
    response_tweet_id
):

    if not response_tweet_id:
        return None

    response_tweet_id = str(
        response_tweet_id
    )

    if response_tweet_id == "nan":
        return None


    # Dataset can contain multiple IDs
    ids = re.split(
        r"[, ]+",
        response_tweet_id
    )


    responses = []

    for response_id in ids:

        response_id = response_id.strip()

        if response_id in tweet_lookup:

            response = tweet_lookup[
                response_id
            ]

            if response:
                responses.append(
                    response
                )


    if responses:

        return " ".join(
            responses
        )


    return None


# ============================================================
# FIND BEST HISTORICAL RESPONSE
# ============================================================

def find_best_response(
    search_results
):

    # First try the strongest matching case
    for result in search_results:

        response = get_historical_response(
            result.get(
                "response_tweet_id",
                ""
            )
        )

        if response:

            return {

                "response": response,

                "similarity": result[
                    "similarity"
                ],

                "source_message": result[
                    "text"
                ]
            }


    return None


# ============================================================
# GENERATE AI-STYLE RESPONSE
# ============================================================

def generate_reply(
    customer_message,
    intent,
    search_results
):

    best_response = find_best_response(
        search_results
    )


    # --------------------------------------------------------
    # Strong historical evidence
    # --------------------------------------------------------

    if best_response:

        historical = (
            best_response["response"]
        )

        return (
            "Based on similar historical "
            "customer-support cases, here is "
            "the most relevant response:\n\n"
            + historical
        )


    # --------------------------------------------------------
    # Fallback when no historical response
    # --------------------------------------------------------

    return (
        "I'm sorry you're experiencing this issue. "
        "Based on the available customer-support data, "
        f"this appears to be a {intent.lower()}. "
        "Please provide any relevant order or account "
        "details so the issue can be investigated."
    )


# ============================================================
# DECIDE ACTION
# ============================================================

def decide_action(
    intent,
    confidence
):

    if confidence >= 0.45:

        return (
            "AUTO-HANDLE",
            "A sufficiently similar historical "
            "customer-support case was found."
        )

    return (
        "ESCALATE",
        "The similarity/confidence is low, "
        "so human review is recommended."
    )


# ============================================================
# COMPLETE AI SUPPORT AGENT
# ============================================================

def support_agent(message):

    # --------------------------------------------------------
    # Search historical data
    # --------------------------------------------------------

    results = search_customer_messages(
        message,
        top_k=5
    )


    if not results:

        return {

            "message": message,

            "intent": "Other",

            "confidence": 0.0,

            "reply": (
                "I'm sorry, but I could not "
                "find a sufficiently similar "
                "case in the support dataset."
            ),

            "action": "ESCALATE",

            "reason": (
                "No historical match was found."
            ),

            "similar_cases": []
        }


    # --------------------------------------------------------
    # Best match
    # --------------------------------------------------------

    best = results[0]

    confidence = best[
        "similarity"
    ]


    # --------------------------------------------------------
    # Intent
    # --------------------------------------------------------

    intent, cluster_id = find_intent(
        best["tweet_id"]
    )


    # --------------------------------------------------------
    # AI-style response
    # --------------------------------------------------------

    reply = generate_reply(
        message,
        intent,
        results
    )


    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    action, reason = decide_action(
        intent,
        confidence
    )


    return {

        "message": message,

        "intent": intent,

        "cluster": cluster_id,

        "confidence": confidence,

        "reply": reply,

        "action": action,

        "reason": reason,

        "similar_cases": results
    }


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("AMAZON CUSTOMER SUPPORT AI AGENT")
    print("=" * 70)

    print(
        "\nThe agent uses historical Amazon "
        "customer-support conversations."
    )


    while True:

        message = input(
            "\nEnter customer message "
            "(type 'exit' to stop): "
        )


        if message.lower().strip() in [
            "exit",
            "quit",
            "q"
        ]:

            print("\nExiting...")
            break


        if not message.strip():

            print(
                "Please enter a customer message."
            )

            continue


        # ====================================================
        # RUN AI AGENT
        # ====================================================

        result = support_agent(
            message
        )


        # ====================================================
        # DISPLAY
        # ====================================================

        print()
        print("-" * 70)

        print(
            "CUSTOMER MESSAGE:"
        )

        print(
            result["message"]
        )


        print(
            "\nPREDICTED INTENT:"
        )

        print(
            result["intent"]
        )


        print(
            "\nINTENT CLUSTER:"
        )

        print(
            result["cluster"]
        )


        print(
            "\nCONFIDENCE:"
        )

        print(
            f"{result['confidence'] * 100:.2f}%"
        )


        print(
            "\nAI RESPONSE:"
        )

        print(
            result["reply"]
        )


        print(
            "\nDECISION:"
        )

        print(
            result["action"]
        )


        print(
            "\nDECISION REASON:"
        )

        print(
            result["reason"]
        )


        # ====================================================
        # SHOW EVIDENCE
        # ====================================================

        print()
        print(
            "HISTORICAL EVIDENCE:"
        )

        print("-" * 70)


        for i, case in enumerate(
            result["similar_cases"],
            1
        ):

            print(
                f"\n{i}. "
                f"Similarity: "
                f"{case['similarity']:.4f}"
            )

            print(
                f"Tweet ID: "
                f"{case['tweet_id']}"
            )

            print(
                f"Customer: "
                f"{case['text'][:300]}"
            )

            response = get_historical_response(
                case.get(
                    "response_tweet_id",
                    ""
                )
            )

            if response:

                print(
                    f"Amazon Response: "
                    f"{response[:500]}"
                )

        print("-" * 70)