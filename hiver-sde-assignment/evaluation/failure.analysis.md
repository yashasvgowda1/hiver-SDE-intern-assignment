# Failure Analysis

## 1. Overview

The AI customer-support system was evaluated using the golden set.
Although the model performs well overall, some incorrect predictions
can occur for ambiguous or closely related customer queries.

## 2. Common Failure Cases

### Case 1: Delivery vs Shipping/Tracking

Example:
"My package hasn't arrived yet."

Expected:
Shipping / Tracking Issue

Possible prediction:
Order / Delivery Issue

Reason:
Both intents contain similar words such as "package", "arrived",
"delivery", and "shipping". TF-IDF similarity may therefore retrieve
a case from the wrong but closely related intent.

---

### Case 2: Refund vs Return

Example:
"I returned my item but haven't received my money."

Expected:
Refund / Return Issue

Possible prediction:
Refund / Return Issue

Reason:
Return and refund requests frequently appear together in the
training data, making the boundary between the two categories
difficult to distinguish.

---

### Case 3: Cancellation vs Delivery

Example:
"Can I cancel my order? It hasn't shipped yet."

Expected:
Cancellation Request

Possible prediction:
Order / Delivery Issue

Reason:
The message contains both cancellation and shipping-related
information. The retrieval model may select a delivery example
because of the shared order/shipping vocabulary.

## 3. Why Failures Occur

The main causes of prediction errors are:

1. Similar vocabulary between different intents.
2. Short customer messages containing very little context.
3. Multiple issues appearing in one customer message.
4. Limited examples for some intents.
5. TF-IDF relies primarily on lexical similarity rather than
   understanding the full semantic meaning of a sentence.

## 4. How the System Handles Uncertainty

The system calculates a similarity/confidence score for each
prediction.

When confidence is low, the system can choose:

ESCALATE

instead of automatically handling the request.

This reduces the risk of giving an incorrect automated response.

## 5. Improvements

Possible future improvements include:

- Using sentence embeddings instead of only TF-IDF.
- Increasing the number of labelled examples.
- Adding more examples for confusing intent pairs.
- Using a stronger semantic retrieval model.
- Adding a human-review step for ambiguous queries.

## 6. Conclusion

The failure analysis shows that most potential errors occur between
closely related customer-support intents rather than completely
unrelated categories. The escalation mechanism provides a fallback
for uncertain predictions while allowing high-confidence cases to
be handled automatically.
