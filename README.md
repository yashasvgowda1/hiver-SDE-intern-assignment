Amazon Customer Support AI Agent
1. Project Overview
This project implements an AI-powered customer support agent that understands customer messages, predicts the customer's intent, retrieves similar historical customer-support conversations, generates a relevant response, provides a confidence score, and decides whether the issue can be automatically handled or should be escalated to a human.

The system is designed as an end-to-end customer-support pipeline using real customer-support datasets.

The main objective is not only to classify a customer message, but to use historical customer-support data to provide a useful and context-aware response.

Example:

Customer: "My order is late."

The system can identify this as an order/delivery issue, retrieve similar historical customer messages, and generate an appropriate support response without inventing information that is not available.

2. Assignment Objective
The system was developed to satisfy the main requirements of the Hiver SDE Intern Assignment:

A runnable repository and pipeline.
A manually labelled golden evaluation set.
An automated evaluation harness.
Baseline comparisons.
Human/LLM-based response-quality evaluation.
Failure analysis with real examples.
Analysis of what can be misleading about the headline metric.
A decision log explaining important engineering decisions.
A discussion of what would be improved with additional development time.
3. Problem Statement
Customer-support systems receive a large number of customer messages.

A useful support AI should be able to:

Understand what the customer is asking.
Identify the customer's intent.
Retrieve similar historical cases.
Generate a useful response.
Estimate prediction confidence.
Automatically handle simple and high-confidence requests.
Escalate uncertain or complex requests.
Avoid making unsupported claims.
The goal of this project is to build a practical customer-support AI pipeline that combines classification, information retrieval, response generation and evaluation.

4. What "Good" Means
For this project, a good customer-support response should satisfy the following criteria.

Relevance
The response should address the customer's actual problem.

Helpfulness
The response should provide a useful next step instead of simply repeating the customer's message.

Context Awareness
The system should use historical customer-support examples when they are relevant.

Accuracy
The response should not invent information such as an order status, delivery date, refund amount or account information.

Clarity
The customer should be able to understand the response easily.

Actionability
The response should explain what the customer can do next or what information is required.

Safe Escalation
If the system is uncertain or the issue requires information that is unavailable, it should be escalated instead of confidently guessing.

5. System Architecture
The complete pipeline is:

Customer Message | v Text Cleaning | v TF-IDF Vectorization | +--------------------------+ | | v v Intent Classification Historical Search | | | v | Similar Customer Cases | | +-------------+------------+ | v Response Generation | v Confidence Estimation | v Auto-Handle / Escalate | v Output

The system therefore combines supervised classification with historical-message retrieval.

6. Dataset
The project uses customer-support datasets stored inside the data/ directory.

The main datasets are:

amazon_customer_messages.csv
amazon_help.csv
amazon_support.csv
intent_clusters.csv
twcs.csv
The manually created evaluation dataset is:

evaluation/golden_set.csv
The datasets are used for different stages of the pipeline.

amazon_support.csv
This dataset provides customer-support messages used by the classification/search pipeline.

amazon_customer_messages.csv
This dataset contains customer messages that can be used for customer-support analysis and retrieval.

amazon_help.csv
This dataset contains support/help-related information.

twcs.csv
This dataset contains Twitter customer-support conversations and provides additional examples of real customer-support language.

intent_clusters.csv
This file associates customer messages with discovered intent clusters.

golden_set.csv
This is the manually labelled evaluation dataset created specifically for evaluating the system.

The golden set is kept separate from the main training/search data to provide an independent evaluation set.

7. Project Structure
hiver-sde-assignment/

data/
    amazon_customer_messages.csv
    amazon_help.csv
    amazon_support.csv
    intent_clusters.csv
    twcs.csv

evaluation/
    golden_set.csv

models/
    intent_model.pkl
    vectorizer.pkl
    search_model.pkl
    messages.pkl

src/
    agent.py
    app.py
    build_index.py
    check_results.py
    create_golden_set.py
    discover_intents.py
    evaluate.py
    extract_brand.py
    inspect_brand.py
    predict.py
    prepare_data.py
    search.py
    train_model.py

README.md
8. Technologies Used
The project is implemented using Python.

Main technologies and libraries:

Python
Pandas
NumPy
Scikit-learn
TF-IDF Vectorizer
Cosine Similarity
Nearest Neighbour Search
Joblib
CSV datasets
Pickle model files
9. Installation
Python 3.10+ is recommended.

Open a terminal in the project root directory.

Install the required Python packages:

pip install pandas numpy scikit-learn joblib
If additional dependencies are required by the local environment, install them using:

pip install -r requirements.txt
The project should be executed from the project root:

hiver-sde-assignment
10. Data Preparation
Run:

python src/prepare_data.py
This step prepares the customer-support data required by the later stages of the pipeline.

The script reads the CSV files from the data/ directory and produces the required prepared output.

11. Intent Discovery
The project contains:

src/discover_intents.py
This script is used to inspect/discover customer-support intent groups from the available customer-support data.

The discovered groups are represented using intent clusters.

12. Intent Categories
The current intent categories used by the system are:

0 - Order / Delivery Issue

1 - Refund / Return Issue

2 - Payment / Billing Issue

3 - Account Issue

4 - Product Issue

5 - Shipping / Tracking Issue

6 - Cancellation Request

7 - Technical / App Issue

8 - General Customer Support

9 - Other

These categories provide a practical customer-support classification structure.

13. Training the Model
The main training script is:

src/train_model.py
Run:

python src/train_model.py
The training pipeline uses customer-support text and the corresponding intent clusters.

The text is converted into numerical features and the classification model is trained.

The trained model is saved inside:

models/
The model achieved approximately 92% accuracy during development on the available evaluation split.

The exact result can vary depending on the data split and environment.

14. Text Representation
TF-IDF is used to convert customer messages into numerical vectors.

TF-IDF was selected because:

It is simple.
It is fast.
It works well for text classification.
It is interpretable.
It performs well for keyword-based customer-support messages.
It is computationally lightweight.
The vectorizer also uses word n-grams so that common phrases can contribute to the representation.

15. Historical Customer Message Retrieval
The project also contains a search/retrieval component.

The search script is:

src/search.py
The search model is built using:

src/build_index.py
Run:

python src/build_index.py
The system converts customer messages into TF-IDF vectors and uses nearest-neighbour search to find similar historical customer messages.

The search result contains:

Tweet ID
Author ID
Customer message
Similarity score
Example:

Customer query:

My order is late.
The system can retrieve historical messages such as:

Not really anything you can do now. It's been in the city all day
and it's still going to be a day late.
The similarity score provides an indication of how closely the historical message matches the customer's query.

16. Prediction
The prediction script is:

src/predict.py
Run:

python src/predict.py
Example:

Enter customer message:
my order is late
The system predicts the relevant intent and confidence.

Example output:

Predicted Intent Cluster: 7
Confidence: 99.9%
The prediction component uses the trained model rather than requiring the user to manually provide an intent.

17. Complete AI Agent
The complete support agent is implemented in:

src/agent.py
Run:

python src/agent.py
The agent combines:

Customer message processing
Intent classification
Confidence estimation
Historical-message retrieval
Response generation
Auto-handle decision
Human escalation
Example:

Enter customer message:
my order is late
The system returns information such as:

INTENT:
Order / Delivery Issue

CONFIDENCE:
High

DRAFT REPLY:
I'm sorry that your order has not arrived as expected.
Please share your order details so the delivery status
can be checked and further assistance can be provided.

DECISION:
AUTO-HANDLE

REASON:
High-confidence routine support issue.
It also displays similar historical customer messages.

18. Response Generation
The response-generation component produces a support response based on the predicted intent.

The response should be useful but should not invent unavailable information.

For example, if the customer says:

My order is late.
The system should not automatically claim:

Your order will arrive tomorrow.
unless an actual delivery estimate exists in the available data.

A safer response is:

I'm sorry that your order has not arrived as expected.
Please share your order details so the delivery status
can be checked and further assistance can be provided.
This prevents hallucination and unsupported promises.

19. Historical Evidence
The system also displays similar historical customer messages.

This provides evidence from the available customer-support dataset.

For example:

Similarity: 0.5012

Historical message:
Not really anything you can do now. It's been in the city all day
and it's still going to be a day late.
The historical examples are not treated as guaranteed facts about the current customer's order.

They are used as supporting examples and context.

20. Confidence and Escalation
The system calculates a confidence/similarity score for its prediction.

High-confidence routine issues can be automatically handled.

Lower-confidence or complex issues can be escalated to a human.

Example:

Intent:
Order / Delivery Issue

Confidence:
0.92

Decision:
AUTO-HANDLE
For uncertain cases:

Decision:
ESCALATE

Reason:
Low confidence or issue may require human investigation.
The purpose of escalation is to prevent the system from confidently answering questions that require information it does not have.

21. Baselines
The assignment requires comparison against simple baselines.

Two useful baselines are defined for comparison.

Baseline 1 - Majority-Class Baseline
The simplest baseline predicts the most common intent for every customer message.

Example:

Every customer message
        |
        v
Most frequent intent
This baseline does not understand the message.

It provides a trivial lower-bound reference for intent classification.

If the proposed model does not outperform this baseline, the model is not providing meaningful classification value.

Baseline 2 - TF-IDF Retrieval Baseline
The second baseline uses only TF-IDF similarity.

Pipeline:

Customer message
        |
        v
    TF-IDF
        |
        v
Nearest historical message
        |
        v
    Retrieved case
This baseline does not perform the complete support-agent workflow.

The proposed system extends this idea by combining:

Intent classification
Historical retrieval
Confidence estimation
Response generation
Auto-handling
Escalation
The retrieval baseline is useful because it shows how much additional value comes from the complete agent compared with simple similarity search.

22. Why Baselines Matter
A model accuracy number by itself is difficult to interpret.

Baselines answer an important question:

Is the proposed system actually better than simpler approaches?
The majority-class baseline provides a trivial reference.

The TF-IDF retrieval baseline provides a simple information-retrieval reference.

The proposed system is evaluated as a complete customer-support pipeline rather than only as a classifier.

23. Golden Evaluation Set
The evaluation dataset is:

evaluation/golden_set.csv
The golden set contains manually labelled customer-support examples.

The purpose of the golden set is to provide an evaluation set that is separate from the training/search data.

The examples should represent different types of customer-support problems, including:

Delivery issues
Shipping issues
Refund requests
Returns
Payment problems
Account problems
Product problems
Cancellation requests
Technical issues
General support requests
The golden set allows the system's predictions to be compared against manually assigned expected labels.

24. Creating the Golden Set
The project includes:

src/create_golden_set.py
This script is used to create/manage the evaluation examples.

The golden set should contain hand-labelled examples and should not simply be generated automatically from the model's predictions.

This prevents the evaluation from becoming circular.

25. Evaluation Harness
The main evaluation script is:

src/evaluate.py
Run:

python src/evaluate.py
Additional result checking is available through:

python src/check_results.py
The evaluation process is designed to measure the system against the golden evaluation data.

Important evaluation measurements include:

Accuracy
Precision
Recall
F1-score
Class-level performance
Response quality
Retrieval relevance
Confidence behaviour
Auto-handle versus escalation behaviour
26. Classification Results
During development, the trained intent classifier achieved approximately:

Accuracy: 92%
The classification report also showed that performance varies between intent classes.

Some classes are easier to distinguish because they contain distinctive vocabulary.

Other classes are more difficult because they contain overlapping words and similar customer language.

The headline accuracy is therefore only one part of the evaluation.

27. Human Evaluation
Automated classification metrics cannot completely measure customer-support quality.

A response can have the correct intent but still be unhelpful.

Human evaluation therefore considers the quality of the generated response.

A human evaluator can score each response using the following criteria.

Relevance
Does the response address the customer's actual issue?

Helpfulness
Does it provide a useful next step?

Accuracy
Does it avoid unsupported claims?

Clarity
Is the response easy to understand?

Safety
Does it avoid inventing order information, delivery dates, refunds or other unsupported facts?

Actionability
Does it tell the customer what to do next?

A simple scoring scale can be used:

1 = Poor
2 = Weak
3 = Acceptable
4 = Good
5 = Excellent
The average score across the golden evaluation examples can then be used as a response-quality measurement.

28. LLM-as-Judge Evaluation
An LLM can also be used as a judge for response quality.

The judge receives:

Customer message
Predicted intent
Generated response
Relevant historical context
The judge evaluates:

Relevance
Helpfulness
Accuracy
Clarity
Safety
The LLM judge should follow a fixed rubric rather than giving an unrestricted opinion.

The judge's result should also be compared with human ratings on a sample of responses.

This helps determine whether the automated judge agrees with human judgement.

29. Human Agreement
LLM-based evaluation should not automatically be treated as ground truth.

A subset of examples should be evaluated by humans.

The human scores can then be compared with the LLM-judge scores.

Agreement can be measured using:

Percentage agreement
Correlation
Agreement by scoring category
If the LLM judge disagrees with humans frequently, its evaluation should not be treated as reliable without further calibration.

30. Failure Analysis
The system has several important failure modes.

Failure Mode 1 - Similar Intents
Example:

Where is my package?
This can potentially belong to:

Order / Delivery Issue
Shipping / Tracking Issue
Both categories contain similar vocabulary.

Hypothesis:

The model needs more discriminative examples or semantic understanding to separate these cases reliably.

Failure Mode 2 - Very Short Messages
Examples:

Help
or:

Still waiting.
These messages contain very little information.

Hypothesis:

There is not enough textual evidence for reliable classification.

Such messages should often result in clarification or escalation.

Failure Mode 3 - Ambiguous Messages
Example:

It still hasn't arrived.
The customer does not explicitly say whether they mean:

An order
A replacement
A refund
A shipment
Another item
Hypothesis:

Conversation history would provide additional context and improve prediction.

Failure Mode 4 - Different Wording
Two customers can describe the same issue differently.

Example:

My order is late.
and:

My parcel hasn't shown up yet.
These messages have similar meanings but may have limited lexical overlap.

Hypothesis:

TF-IDF is based heavily on word overlap.

A semantic embedding model would improve retrieval and classification for paraphrased customer messages.

Failure Mode 5 - Misleading Confidence
The system can sometimes produce a high confidence value for an incorrect or ambiguous prediction.

For example:

Confidence: 99%
does not mean that the system is guaranteed to be correct.

The score represents the model's confidence/similarity based on its learned representation.

It is not a guarantee of real-world correctness.

Hypothesis:

Confidence calibration and better uncertainty estimation are required for a production system.

31. What Is Misleading About the Headline Number?
The approximately 92% accuracy result is useful, but it should not be interpreted as:

"92% of real customer problems will be solved successfully."
There are several reasons.

First, accuracy measures classification correctness, not response quality.

Second, some classes contain many more examples than others.

Third, a model can correctly classify an issue but still generate an unhelpful response.

Fourth, a retrieved historical message may be similar in wording but not actually provide the correct solution.

Fifth, confidence scores are not guaranteed probabilities of real-world correctness.

Sixth, real customers may use language that is not represented in the evaluation set.

Therefore, the headline accuracy should be considered only one measurement.

A production-quality evaluation should also consider:

Macro F1
Per-class recall
Retrieval relevance
Response quality
Human agreement
Escalation rate
Calibration
Failure cases
32. Safety and Hallucination Control
The system should not invent information.

For example, if the customer says:

My order is late.
The system should not claim:

Your order will arrive in two days.
unless a trusted order-status system actually provides that information.

Instead, it should communicate uncertainty and request the information required to check the issue.

Historical customer messages are evidence for similarity, not proof of the current customer's order status.

This distinction is important for a real customer-support system.

33. Auto-Handle versus Escalation
The system supports two high-level decisions.

AUTO-HANDLE
Used for high-confidence routine cases that can be answered safely.

ESCALATE
Used when:

Confidence is low.
The customer message is ambiguous.
The issue requires unavailable account/order information.
The issue may require human investigation.
The system cannot safely provide a definitive answer.
This design prioritizes safe handling over blindly maximizing automation.

34. Reproducible Pipeline
The main pipeline can be reproduced using:

python src/prepare_data.py

python src/train_model.py

python src/build_index.py

python src/predict.py

python src/agent.py

python src/evaluate.py
The scripts are separated so that each stage can be independently tested and improved.

35. Important Source Files
prepare_data.py
Prepares the customer-support data.

discover_intents.py
Helps discover/inspect intent clusters.

train_model.py
Trains the intent classification model.

build_index.py
Builds the historical-message search index.

search.py
Searches for similar customer messages.

predict.py
Predicts the intent of a new customer message.

agent.py
Runs the complete AI customer-support workflow.

evaluate.py
Evaluates the system on the evaluation data.

check_results.py
Checks evaluation results.

create_golden_set.py
Creates/manages the manually labelled evaluation set.

app.py
Provides the application entry point/interface.

extract_brand.py
Extracts brand-related information from the available data.

inspect_brand.py
Inspects brand-related data.

36. Decision Log
The following are important engineering decisions made during development.

Decision 1 - Use TF-IDF
TF-IDF was selected because it is lightweight, fast and interpretable for customer-support text.

Decision 2 - Use historical customer messages
Historical customer messages provide real examples of how customers describe problems.

Decision 3 - Keep the golden set separate
The golden set is separated from the main data so that evaluation is less likely to suffer from data leakage.

Decision 4 - Use intent clusters
Intent clusters provide a practical way to group customer-support issues.

Decision 5 - Add historical retrieval
Retrieval provides supporting evidence and makes the system more useful than a fixed-response classifier.

Decision 6 - Add confidence estimation
Confidence helps distinguish high-confidence cases from uncertain cases.

Decision 7 - Add escalation
Some customer issues require human investigation or information unavailable to the AI.

Decision 8 - Avoid unsupported promises
The system should not invent delivery dates, refund amounts, order status or other information.

Decision 9 - Evaluate response quality separately
Intent accuracy alone cannot measure customer-support response quality.

Decision 10 - Keep components modular
Separate scripts make the system easier to debug, reproduce and improve.

Decision 11 - Use simple baselines
Baseline comparisons make it possible to determine whether the proposed approach provides value beyond trivial or simple approaches.

Decision 12 - Use human evaluation
Human evaluation is necessary because response usefulness cannot be completely measured using classification accuracy.

37. Limitations
The current system has several limitations.

Limited semantic understanding
TF-IDF primarily captures lexical similarity and may struggle with paraphrases.

Limited conversation context
A single customer message may not contain enough information to determine the correct intent.

No direct live order system
The system does not directly access a customer's live order status.

No guaranteed delivery prediction
The system cannot reliably predict an actual delivery date unless such information is provided by a trusted data source.

Confidence is not perfectly calibrated
A high confidence score does not necessarily mean that the prediction is correct.

Response generation is limited
The current response generation uses predefined support logic rather than a fully generative production LLM.

Dataset limitations
The training and evaluation results depend on the available datasets and labels.
38. Example End-to-End Workflow
Input:

My order is late.
Step 1:

The message is cleaned and converted into a TF-IDF representation.

Step 2:

The trained classifier predicts the most likely intent.

Step 3:

The system estimates prediction confidence.

Step 4:

Historical customer messages are searched.

Step 5:

Similar historical examples are retrieved.

Step 6:

The system generates a suitable customer-support response.

Step 7:

The system decides whether the issue can be auto-handled or should be escalated.

Output:

Intent:
Order / Delivery Issue

Confidence:
High

Response:
I'm sorry that your order has not arrived as expected.
Please share your order details so the delivery status
can be checked and further assistance can be provided.

Decision:
AUTO-HANDLE

Historical Evidence:
Similar customer-support messages are displayed separately.
39. Conclusion
This project implements an end-to-end customer-support AI agent using real customer-support datasets.

The system combines:

Data preparation
Intent discovery
Intent classification
TF-IDF text representation
Historical customer-message retrieval
Similarity search
Response generation
Confidence estimation
Automatic handling
Human escalation
Golden-set evaluation
Baseline comparison
Human/LLM response evaluation
Failure analysis
The intent classification model achieved approximately 92% accuracy during development.

However, this number is not treated as the complete measure of system quality.

A practical customer-support AI must also be evaluated on response usefulness, retrieval relevance, safety, uncertainty, human agreement and appropriate escalation.

The project therefore focuses on building a reproducible and practical customer-support AI pipeline rather than optimizing only for a single accuracy number.
