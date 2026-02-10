📧 Spam Email Detector Using Machine Learning

A Machine Learning-based web application that detects whether an email or message is SPAM or NOT SPAM using Natural Language Processing (NLP).
The model is trained on a real-world dataset and deployed using Gradio on Hugging Face Spaces.

🚀 Features

Real-time spam detection

Spam probability score

Explainable predictions

Colorful and user-friendly UI

Machine Learning-based classification

Hugging Face deployment ready

📊 Dataset

This project uses the UCI SMS Spam Collection Dataset, also available on Kaggle.

Dataset Details

Total messages: 5,500+

Classes: Spam / Ham

Real-world SMS & email samples

Language: English

Sources

UCI Machine Learning Repository:
https://archive.ics.uci.edu/ml/datasets/sms+spam+collection

Kaggle Dataset:
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

🧠 Model Training & Fine-Tuning

The model is trained and optimized using the following steps:

Data Cleaning

Removing punctuation

Lowercasing text

Removing stopwords

Text Vectorization

TF-IDF (Term Frequency–Inverse Document Frequency)

Dataset Splitting

80% Training

20% Testing

Model Training

Logistic Regression / Naive Bayes

Fine-Tuning

Hyperparameter tuning

Cross-validation

Feature optimization

Evaluation

Accuracy

Precision

Recall

F1-Score

Model Saving

Using Joblib for reuse

Reference (Scikit-learn):
https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

🛠 Tech Stack

Python

Pandas

NumPy

Scikit-learn

NLP (TF-IDF)

Gradio

Joblib

Hugging Face Spaces

Libraries Documentation:

Scikit-learn: https://scikit-learn.org/stable/

Gradio: https://www.gradio.app/

Hugging Face Spaces: https://huggingface.co/docs/hub/spaces

📁 Project Structure
spam-email-detector/
│
├── app.py
├── train_model.py
├── spam_model.pkl
├── spam_vectorizer.pkl
├── requirements.txt
└── README.md

▶️ How to Run Locally
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run the App
python app.py


The application will run on:

http://localhost:7860

🌐 Deployment

This project is deployed on Hugging Face Spaces using Gradio.

Deployment Guide:
https://huggingface.co/docs/hub/spaces-sdks-gradio

Live Demo:
(https://areebach-spamemail.hf.space)

📈 Results

Before Fine-Tuning:

Accuracy: ~95%

After Fine-Tuning:

Improved precision and recall

Reduced false positives

Better generalization

Model evaluation is performed using Scikit-learn metrics.

Reference:
https://scikit-learn.org/stable/modules/model_evaluation.html

🧪 Testing

Users can test by entering:

Promotional messages

Lottery messages

Normal emails

Business messages

The system returns:

Prediction (SPAM / NOT SPAM)

Probability Score

Explanation

👩‍💻 Author

Areeba Chaudhry
Software Engineer | Machine Learning Enthusiast

📜 License

This project is for educational and research purposes.

📚 References

UCI Dataset
https://archive.ics.uci.edu/ml/datasets/sms+spam+collection

Kaggle Dataset
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Scikit-learn Documentation
https://scikit-learn.org/stable/

Gradio Documentation
https://www.gradio.app/docs

Hugging Face Spaces
https://huggingface.co/docs/hub/spaces
