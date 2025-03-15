# **🛡️ Toxicity Comment Classification using LSTM**  

<img width="631" alt="image" src="https://github.com/user-attachments/assets/95d49282-e867-4d62-992c-4638a7f78966" />


![Toxicity Classification](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*ZsMZ2ACkgrusXxTtPVXk9A.png)  
*(Image Source: Medium - Toxicity Detection in NLP)*  

## **📌 Overview**  
Toxicity Comment Classification is a deep learning-based project that classifies user comments into multiple toxicity categories using a **Bidirectional LSTM** model.  

🚀 **Key Features:**  
✔️ Uses **TextVectorization** for preprocessing.  
✔️ Implements a **Bidirectional LSTM** network.  
✔️ Multi-label classification with **sigmoid activation**.  
✔️ Built with **TensorFlow/Keras** and **Dockerized** for deployment.  

---

## **🛠️ Model Architecture**  

🔹 The model consists of an **Embedding Layer**, a **Bidirectional LSTM**, and **Dense layers** with ReLU activations.  
🔹 The final layer has **six outputs**, each corresponding to a different toxicity category.  

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

model = Sequential([
    Embedding(input_dim=MAX_FEATURES + 1, output_dim=32),
    Bidirectional(LSTM(32, activation='tanh')),
    Dense(128, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(6, activation='sigmoid')
])

model.build(input_shape=(None, None))
```

📊 **Architecture Summary:**  

| Layer               | Type               | Activation |
|---------------------|--------------------|------------|
| Embedding          | 32-Dimensional     | -          |
| Bidirectional LSTM | 32 Units           | Tanh       |
| Dense             | 128 Units          | ReLU       |
| Dense             | 256 Units          | ReLU       |
| Dense             | 128 Units          | ReLU       |
| Output Layer      | 6 Units (Multi-Label) | Sigmoid    |

---

## **📖 Data Preprocessing**  

📌 **TextVectorization** is used to tokenize and convert text data into numerical sequences before passing it to the model.  

```python
from tensorflow.keras.layers import TextVectorization

vectorizer = TextVectorization(
    max_tokens=MAX_FEATURES, 
    output_sequence_length=1800, 
    output_mode='int'
)
```

📝 **Preprocessing Workflow:**  
1️⃣ **Raw Text Input** → 2️⃣ **Text Cleaning** → 3️⃣ **Tokenization (TextVectorization)** → 4️⃣ **Model Input**  

---

## **⚡ Installation & Setup**  

### **🔹 Clone the Repository**  
```bash
git clone https://github.com/your-username/toxicity-classification-lstm.git
cd toxicity-classification-lstm
```

### **🔹 Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **🔹 Train the Model**  
```bash
python train.py
```

---

## **🐳 Running with Docker**  

### **🔹 Build the Docker Image**  
```bash
docker build -t toxicity-lstm .
```

### **🔹 Run the Docker Container**  
```bash
docker run -it --rm -p 8888:8888 -v $(pwd):/workspace toxicity-lstm
```
✅ **Access Jupyter Notebook at:** `http://localhost:8888`  

---

## **📊 Dataset**  
📌 The dataset contains user comments labeled with multiple toxicity categories such as:  
✔️ **Toxic**  
✔️ **Severe Toxic**  
✔️ **Obscene**  
✔️ **Threat**  
✔️ **Insult**  
✔️ **Identity Hate**  

📂 The dataset is preprocessed using **TextVectorization**, tokenized, and then passed into the **LSTM model** for classification.  

---

## **🎯 Model Performance (Example Metrics)**  

| Metric            | Score  |
|------------------|--------|
| Precision       | 92.5%  |
| Recall         | 89.8%  |
| F1-Score      | 91.1%  |

*(Metrics may vary based on dataset and training configuration.)*  

📉 **Loss & Accuracy Curve** *(Sample Visualization)*:  
![Training Graph](https://raw.githubusercontent.com/jbrownlee/Datasets/master/lstm_training_graph.png)  

---

## **🔮 Future Enhancements**  
✅ **Hyperparameter tuning** for improved model performance.  
✅ **Integration with Gradio** for interactive model testing.  
✅ **Deployment as a REST API** or Web App.  

---

## **📜 License**  
This project is licensed under the **MIT License**.

---

## **🤝 Contributions**  
Contributions are always welcome! 🚀  

If you'd like to contribute:  
1️⃣ **Fork the repository**  
2️⃣ **Make your changes**  
3️⃣ **Submit a Pull Request (PR)**  

---

### **🎉 Thank You for Checking Out This Project!** 🚀  

![NLP GIF](https://media3.giphy.com/media/xT9IguC6bxSeM2F7oU/giphy.gif)  

