# Sentiment Analysis on Amazon Fine Food Reviews

## Project Overview
This project aims to perform sentiment analysis on a dataset of Amazon Fine Food product reviews using machine learning techniques. Sentiment analysis is a critical task in natural language processing (NLP) that helps categorize text into sentiments such as **positive**, **neutral**, or **negative**. By applying machine learning models, this project provides valuable insights into customer feedback, which businesses can leverage to improve their offerings and better understand consumer behavior.

---

## Dataset Information
The dataset, sourced from [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews), contains:

- **Number of Reviews**: 568,000+
- **Time Period**: October 1999 to October 2012
- **Number of Users**: 256,059
- **Number of Products**: 74,258
- **Size**: 300.9 MB

### Features in the Dataset:
- **Id**: Unique identifier for each review
- **ProductId**: Identifier for the product being reviewed
- **UserId**: Identifier for the user who wrote the review
- **ProfileName**: Name of the user profile
- **HelpfulnessNumerator**: Number of users who found the review helpful
- **HelpfulnessDenominator**: Total number of users who voted on the helpfulness
- **Score**: Rating given by the user (1 to 5)
- **Time**: Timestamp of the review
- **Summary**: Short summary of the review
- **Text**: Full text of the review

The dataset's richness in text data and diversity of products reviewed make it ideal for sentiment analysis.

---

## Key Components
### 1. Data Handling & Preprocessing
#### Steps Involved:
- **Data Cleaning**:
  - Removed irrelevant columns (e.g., `ProfileName`).
  - Filled missing values in the `Summary` column with "No Summary".
- **Data Transformation**:
  - Derived a `Sentiment` column based on the `Score` feature:
    - `Score > 3`: Positive
    - `Score = 3`: Neutral
    - `Score < 3`: Negative
- **Text Preprocessing**:
  - Stripped URLs, HTML tags, special characters, and numbers.
  - Tokenized, lemmatized, and removed stopwords using the NLTK library.

### 2. Data Visualization
Exploratory data analysis (EDA) included:
- **Word Cloud**: Visualized frequent terms in reviews.
- **Score Distribution**: Displayed the distribution of review ratings.
- **Helpfulness vs. Rating**: Analyzed the relationship between helpfulness and ratings.
- **Multivariate Visualization**: Explored interactions between sentiment, helpfulness, and ratings.

### 3. Data Storage
SQLite was chosen for its simplicity and efficiency in handling large datasets. The following tasks were performed:
- Created two tables (`reviews` and `transformed_reviews`) to store raw and processed data.
- Used SQL queries for tasks such as counting records, identifying duplicates, and summarizing sentiments.

### 4. Handling Data Imbalance
The dataset was highly imbalanced, with most reviews labeled as positive. Random undersampling was used to balance the dataset across all sentiment classes, ensuring the model learned effectively.

### 5. Feature Extraction
TF-IDF (Term Frequency-Inverse Document Frequency) was used to convert textual data into numerical features. The top 3,000 most relevant words were selected, optimizing the model's performance.

### 6. Model Development
#### Models Used:
1. **Logistic Regression**:
   - Simplicity and interpretability.
   - Effective for linearly separable data.

2. **Random Forest**:
   - Captured complex patterns and non-linear relationships.
   - Reduced overfitting through ensemble techniques.

#### Model Performance:
| Metric               | Logistic Regression | Random Forest |
|----------------------|---------------------|---------------|
| **Accuracy**         | 71%                | 75%           |
| **Precision**        | 71%                | 75%           |
| **Recall**           | 71%                | 75%           |
| **F1-Score**         | 71%                | 75%           |

### 7. Hyperparameter Tuning
- **Logistic Regression**: Tuned using `GridSearchCV`.
- **Random Forest**: Tuned using `RandomizedSearchCV`.

### 8. Deployment
The sentiment analysis functionality was deployed using a user-friendly web application built with [Streamlit](https://streamlit.io/). Users can input reviews and receive real-time sentiment predictions.
- App Link: [Sentiment Analysis App](https://sentimentanalysisapp-9.streamlit.app/review)

---

## Repository Structure
```
├── data/
│   ├── raw_data.csv        # Original dataset
│   ├── processed_data.csv  # Preprocessed data
├── notebooks/
│   ├── eda.ipynb           # Exploratory Data Analysis
│   ├── model_building.ipynb # Model Training and Evaluation
├── models/
│   ├── logistic_model.pkl  # Logistic Regression model
│   ├── random_forest.pkl   # Random Forest model
├── app/
│   ├── app.py              # Streamlit application
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
```

---

## Getting Started

### Prerequisites
- Python 3.8+
- Libraries: NLTK, Pandas, NumPy, Scikit-learn, Streamlit, SQLite3

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/amazon-sentiment-analysis.git
   ```
2. Navigate to the project directory:
   ```bash
   cd amazon-sentiment-analysis
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
Run the Streamlit app:
```bash
streamlit run app/app.py
```

---

## Results and Insights
- **Best Performing Model**: Random Forest
  - **Accuracy**: 75%
  - Performed well across all sentiment classes, especially for positive reviews.
- **Challenges**:
  - Neutral sentiment classification required further tuning.
- **Future Work**:
  - Explore deep learning models such as RNNs or Transformers.
  - Implement advanced feature extraction methods like Word2Vec or GloVe.
  - Enhance data augmentation for improved generalization.

---

## Acknowledgments
- Dataset: [Kaggle - Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- Tools: Python, Streamlit, Scikit-learn, NLTK, SQLite

---

## License
This project is licensed under the [MIT License](LICENSE).

---


