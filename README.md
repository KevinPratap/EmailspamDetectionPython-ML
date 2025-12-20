# 📧 Spam SMS Classifier

This project develops and fine-tunes a Machine Learning model to classify SMS messages as either 'ham' (not spam) or 'spam'. It demonstrates the full lifecycle from data loading and cleaning to model training, evaluation, and saving, along with insights into the model's performance and potential real-world integration.

--- 

## ✨ Key Features

*   **Data Loading & Preprocessing**: Efficiently loads SMS data and performs text cleaning (lowercase, alphanumeric filtering).
*   **Robust Filtering**: Implements effective methods to handle and remove empty or whitespace-only messages.
*   **TF-IDF Vectorization**: Transforms raw text messages into numerical feature vectors.
*   **Multinomial Naive Bayes Model**: Trains a robust classification model for spam detection.
*   **Model Evaluation**: Provides comprehensive evaluation metrics including accuracy, precision, recall, and F1-score.
*   **Model Fine-tuning**: Explores hyperparameter tuning for `TfidfVectorizer` to optimize model performance, especially spam recall.
*   **Model Persistence**: Saves the trained vectorizer and model for future use.
*   **Interactive Prediction**: Allows users to input new messages and get instant spam/ham predictions.
*   **Visualizations**: Includes plots for data distribution, confusion matrix, and top indicative features for both spam and ham.

--- 

## 🛠️ Technologies Used

*   **Python** 🐍
*   **Pandas**: For data manipulation and analysis.
*   **NLTK**: For natural language processing tasks.
*   **Scikit-learn**: For machine learning models (TF-IDF Vectorizer, Multinomial Naive Bayes).
*   **Joblib**: For saving and loading Python objects (models, vectorizers).
*   **Matplotlib** & **Seaborn**: For data visualization.

--- 

## 🚀 How to Use

1.  **Clone the Repository (or open in Colab)**
    (Assuming this notebook is converted to a script or the user has access to the `.ipynb` file and `spam.csv`)

2.  **Install Dependencies**
    ```bash
    pip install pandas scikit-learn nltk matplotlib seaborn joblib
    ```

3.  **Data Preparation**
    *   Ensure `spam.csv` is available in your working directory or adjust the path in the code.

4.  **Run the Notebook**
    Execute the cells sequentially to:
    *   Load and clean the data.
    *   Train the spam classification model.
    *   Evaluate its performance.
    *   Save the `TfidfVectorizer` and `MultinomialNB` model.

5.  **Make Predictions**
    The notebook includes a `predict_spam_fine_tuned` function. After running the training cells, you can load the saved model and vectorizer to predict on new messages:
    
    ```python
    import joblib
    
    # Assuming 'tfidf_vectorizer_fine_tuned.joblib' and 'spam_classifier_model_fine_tuned.joblib' are saved
    loaded_vectorizer_fine_tuned = joblib.load('tfidf_vectorizer_fine_tuned.joblib')
    loaded_model_fine_tuned = joblib.load('spam_classifier_model_fine_tuned.joblib')

    # The clean_text function is defined in the notebook's training section
    def clean_text(text):
        text = str(text).lower()
        allowed_punct = {"'", "’"}
        text = ''.join([char for char in text if char.isalnum() or char.isspace() or char in allowed_punct])
        return text

    def predict_spam_fine_tuned(text):
        text_clean = clean_text(text)
        text_vec = loaded_vectorizer_fine_tuned.transform([text_clean])
        pred = loaded_model_fine_tuned.predict(text_vec)
        return 'Spam' if pred[0] == 1 else 'Not Spam'

    # Example prediction
    message = "Congratulations! You've won a free prize. Claim it now!"
    print(f"Message: '{message}' -> Prediction: {predict_spam_fine_tuned(message)}")
    ```

--- 

## 📈 Model Performance (Fine-tuned)

The fine-tuned model achieved the following results:

*   **Accuracy**: 0.9776
*   **Spam Recall**: 0.85
*   **Spam Precision**: 1.00

**Confusion Matrix (Sample Run)**:

|                   | Predicted Ham | Predicted Spam |
| :---------------- | :------------ | :------------- |
| **Actual Ham**    | 947           | 0              |
| **Actual Spam**   | 25            | 142            |

This indicates that the model is very good at identifying ham messages and avoids misclassifying them as spam (0 False Positives). While it correctly identifies a large portion of spam (142 True Positives), a small number of spam messages (25 False Negatives) are still missed.

### Top Features Indicative of Spam:
`to`, `call`, `free`, `your`, `for`, `txt`, `or`, `you`, `now`, `text`, `the`, `mobile`, `from`, `stop`, `claim`, `ur`, `on`, `is`, `have`, `with`

### Top Features Indicative of Ham:
`you`, `to`, `the`, `in`, `me`, `is`, `and`, `my`, `it`, `that`, `ok`, `of`, `for`, `are`, `not`, `at`, `so`, `but`, `have`, `can`

--- 

## 💡 Development Insights

*   **Data Source Issue**: Initially, the `ucirvine/sms_spam` dataset caused persistent filtering issues. Switching to a local `spam.csv` resolved these problems, highlighting the importance of reliable data sources and robust data loading.
*   **Filtering Empty Messages**: Crucial for model stability; a `lambda x: x.strip() != ''` approach proved effective.
*   **TF-IDF Parameter Tuning**: Adjusting `ngram_range` and `min_df` significantly impacted spam recall. The combination of `ngram_range=(1, 2)` and `min_df=0.001` yielded the best results, improving spam detection without increasing false positives.

--- 

## ☁️ Future Work & Automated Email Processing (Conceptual)

Integrating this model into a real-world automated email processing system would involve several external components:

*   **Email Retrieval**: Using `imaplib` or `poplib` to access mail servers.
*   **Email Parsing**: Extracting text content from various email formats.
*   **Scheduling**: Automating the process to run periodically (e.g., with `cron`, `APScheduler`).
*   **Subsequent Actions**: Moving emails to spam folders, tagging, or flagging based on model predictions.

This notebook lays the foundation, but a full system requires significant engineering beyond the notebook environment. 🚀
