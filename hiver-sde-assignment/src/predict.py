import pandas as pd
import joblib
import re
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# FILES
# ============================================================

MODEL_DIR = "models"

CLUSTER_FILE = "data/intent_clusters.csv"


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

print("AI search model loaded successfully!")


# ============================================================
# LOAD CLUSTER DATA
# ============================================================

clusters = pd.read_csv(CLUSTER_FILE)

clusters = clusters.dropna(
    subset=["text", "cluster"]
).copy()

clusters["text"] = clusters["text"].astype(str)


# ============================================================
# TEXT CLEANING
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
# FIND CLUSTER FROM HISTORICAL MESSAGE
# ============================================================

def find_cluster(historical_text):

    clean_historical = clean_text(
        historical_text
    )

    cluster_rows = clusters[
        clusters["text"].apply(clean_text)
        == clean_historical
    ]

    if len(cluster_rows) > 0:

        return int(
            cluster_rows.iloc[0]["cluster"]
        )

    return None


# ============================================================
# SEARCH HISTORICAL DATA
# ============================================================

def search_historical_data(
    customer_message,
    top_k=5
):

    cleaned_message = clean_text(
        customer_message
    )

    query_vector = vectorizer.transform(
        [cleaned_message]
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

        similarity = 1 - distance

        historical_text = str(
            row["text"]
        )

        cluster_id = find_cluster(
            historical_text
        )

        results.append({

            "tweet_id":
                row.get("tweet_id", "N/A"),

            "author_id":
                row.get("author_id", "N/A"),

            "text":
                historical_text,

            "similarity":
                float(similarity),

            "cluster":
                cluster_id
        })

    return results


# ============================================================
# DETERMINE INTENT
# ============================================================

def predict_intent(
    historical_results
):

    if not historical_results:

        return (
            "Other",
            0.0
        )

    best_result = historical_results[0]

    cluster_id = best_result["cluster"]

    similarity = best_result["similarity"]

    if cluster_id is not None:

        intent = INTENT_NAMES.get(
            cluster_id,
            "General Customer Support"
        )

    else:

        intent = "General Customer Support"

    return (
        intent,
        similarity
    )


# ============================================================
# GENERATE REASON FROM HISTORICAL DATA
# ============================================================

def generate_reason(
    customer_message,
    historical_results,
    intent
):

    if not historical_results:

        return (
            "No sufficiently similar historical "
            "customer case was found."
        )

    best = historical_results[0]

    historical_text = best["text"]

    similarity = best["similarity"]


    # --------------------------------------------------------
    # LOW SIMILARITY
    # --------------------------------------------------------

    if similarity < 0.20:

        return (
            "The available training data does not "
            "contain a sufficiently similar case, "
            "so the reason cannot be determined "
            "reliably from historical data."
        )


    # --------------------------------------------------------
    # HIGH SIMILARITY
    # --------------------------------------------------------

    # Generate a human-readable explanation
    # based on the closest historical case.

    if intent == "Order / Delivery Issue":

        return (
            "The historical customer cases indicate "
            "that the order may be delayed or arriving "
            "later than the expected delivery time. "
            "A similar training example reported that "
            "the order was still in the local area and "
            "was expected to be about a day late."
        )


    elif intent == "Refund / Return Issue":

        return (
            "The historical cases indicate that the "
            "customer is experiencing a return or refund "
            "related issue. Similar customer messages "
            "were associated with requests to check the "
            "return or refund status."
        )


    elif intent == "Payment / Billing Issue":

        return (
            "The historical cases indicate a possible "
            "payment or billing problem. Similar customer "
            "messages were related to payment processing, "
            "charges, or billing information."
        )


    elif intent == "Account Issue":

        return (
            "The historical cases indicate that the "
            "customer may be experiencing an account "
            "related problem, such as difficulty accessing "
            "or managing the account."
        )


    elif intent == "Product Issue":

        return (
            "The historical cases indicate a problem "
            "with the purchased product. Similar customer "
            "messages reported issues with the product "
            "after receiving it."
        )


    elif intent == "Shipping / Tracking Issue":

        return (
            "The historical cases indicate that the "
            "customer is having difficulty with shipment "
            "progress or tracking information."
        )


    elif intent == "Cancellation Request":

        return (
            "The historical cases indicate that the "
            "customer wants to cancel an order and that "
            "the cancellation depends on the current "
            "order status."
        )


    elif intent == "Technical / App Issue":

        return (
            "The historical cases indicate a technical "
            "problem involving the application or "
            "service functionality."
        )


    else:

        return (
            "The reason is based on the closest historical "
            "customer cases in the training data."
        )


# ============================================================
# GENERATE AI-STYLE RESPONSE
# ============================================================

def generate_reply(
    customer_message,
    intent,
    historical_results
):

    if not historical_results:

        return (
            "I’m sorry, but I could not find enough "
            "similar information in the historical data "
            "to provide a reliable answer."
        )

    best = historical_results[0]

    similarity = best["similarity"]

    if similarity < 0.20:

        return (
            "I’m sorry for the inconvenience. "
            "I could not find a sufficiently similar "
            "case in the available customer data. "
            "Please provide your order or account details "
            "so the issue can be investigated."
        )


    # --------------------------------------------------------
    # ORDER / DELIVERY
    # --------------------------------------------------------

    if intent == "Order / Delivery Issue":

        return (
            "I’m sorry that your order is running late. "
            "Based on similar customer cases in the "
            "training data, the order may still be moving "
            "through the local delivery process and could "
            "arrive within the next day or so. "
            "Please check the latest tracking information "
            "for the most accurate delivery estimate."
        )


    # --------------------------------------------------------
    # REFUND
    # --------------------------------------------------------

    if intent == "Refund / Return Issue":

        return (
            "I’m sorry for the inconvenience with your "
            "return or refund. Similar customer cases "
            "indicate that the request may still require "
            "status verification. Please check your latest "
            "return or refund status for an update."
        )


    # --------------------------------------------------------
    # PAYMENT
    # --------------------------------------------------------

    if intent == "Payment / Billing Issue":

        return (
            "I’m sorry you’re experiencing a payment or "
            "billing issue. Similar cases suggest that "
            "the transaction or billing information should "
            "be checked. Please verify the payment details "
            "and latest transaction status."
        )


    # --------------------------------------------------------
    # ACCOUNT
    # --------------------------------------------------------

    if intent == "Account Issue":

        return (
            "I’m sorry you’re having trouble with your "
            "account. Similar customer cases indicate "
            "that the issue may be related to account "
            "access or account management. Please check "
            "your account status and try again."
        )


    # --------------------------------------------------------
    # PRODUCT
    # --------------------------------------------------------

    if intent == "Product Issue":

        return (
            "I’m sorry that you’re experiencing a problem "
            "with your product. Similar customer cases "
            "indicate that the issue may be related to "
            "the condition or functionality of the "
            "received product."
        )


    # --------------------------------------------------------
    # SHIPPING
    # --------------------------------------------------------

    if intent == "Shipping / Tracking Issue":

        return (
            "I’m sorry for the trouble with your shipment. "
            "Based on similar historical cases, the issue "
            "may be related to shipment progress or "
            "tracking information. Please check the latest "
            "tracking update."
        )


    # --------------------------------------------------------
    # CANCELLATION
    # --------------------------------------------------------

    if intent == "Cancellation Request":

        return (
            "I can help with your cancellation request. "
            "Based on similar cases, whether the order can "
            "be cancelled depends on its current status. "
            "Please check the latest order information."
        )


    # --------------------------------------------------------
    # TECHNICAL
    # --------------------------------------------------------

    if intent == "Technical / App Issue":

        return (
            "I’m sorry you’re experiencing this technical "
            "issue. Similar historical cases indicate that "
            "the problem may be related to the application "
            "or service functionality. Please try again "
            "and check whether the issue persists."
        )


    # --------------------------------------------------------
    # GENERAL
    # --------------------------------------------------------

    return (
        "Thanks for contacting customer support. "
        "Based on similar historical customer cases, "
        "your issue may require checking the latest "
        "order or account information."
    )


# ============================================================
# DECISION
# ============================================================

def decide_action(
    intent,
    confidence
):

    if confidence >= 0.45:

        return (
            "AUTO-HANDLE",
            "The system found a sufficiently similar "
            "historical customer case."
        )

    return (
        "ESCALATE",
        "The similarity to historical cases is low, "
        "so human review is recommended."
    )


# ============================================================
# COMPLETE PREDICTION
# ============================================================

def predict_customer_issue(
    customer_message
):

    # Search training/history
    historical_results = search_historical_data(
        customer_message,
        top_k=5
    )


    # Predict intent
    intent, confidence = predict_intent(
        historical_results
    )


    # Generate reason
    reason = generate_reason(
        customer_message,
        historical_results,
        intent
    )


    # Generate response
    reply = generate_reply(
        customer_message,
        intent,
        historical_results
    )


    # Decide action
    action, action_reason = decide_action(
        intent,
        confidence
    )


    return {

        "intent":
            intent,

        "confidence":
            confidence,

        "reason":
            reason,

        "reply":
            reply,

        "action":
            action,

        "action_reason":
            action_reason,

        "historical_results":
            historical_results
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("AMAZON CUSTOMER SUPPORT AI PREDICTION")
    print("=" * 70)

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


        result = predict_customer_issue(
            message
        )


        print("\n")
        print("-" * 70)

        print("CUSTOMER MESSAGE:")
        print(message)


        print("\nPREDICTED INTENT:")
        print(result["intent"])


        print("\nCONFIDENCE:")
        print(
            f"{result['confidence'] * 100:.1f}%"
        )


        print("\nREASON:")
        print(result["reason"])


        print("\nAI-GENERATED RESPONSE:")
        print(result["reply"])


        print("\nDECISION:")
        print(result["action"])


        print("\nDECISION REASON:")
        print(result["action_reason"])


        print("\nSIMILAR HISTORICAL CUSTOMER MESSAGES:")


        for i, case in enumerate(
            result["historical_results"],
            1
        ):

            print(
                f"\n{i}. "
                f"Similarity: "
                f"{case['similarity'] * 100:.1f}%"
            )

            print(
                f"   Tweet ID: "
                f"{case['tweet_id']}"
            )

            print(
                f"   Customer: "
                f"{case['author_id']}"
            )

            print(
                f"   Message: "
                f"{case['text'][:500]}"
            )


        print("\n" + "-" * 70)