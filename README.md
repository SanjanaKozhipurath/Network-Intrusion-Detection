# 🛡️ Network Intrusion Detection System

## Project Overview
This project implements a **Network Intrusion Detection System (NIDS)** using the **NSL-KDD dataset**.  
It combines:

- Classification models (Random Forest, Logistic Regression)  
- Anomaly Detection (Isolation Forest)  
- Hybrid Models (RF + IF, LR + IF)  
- Distributed training using Apache Spark  

The system is built with an interactive **Streamlit UI** that guides the user through the full ML pipeline.

---

## Objectives

- Detect malicious network traffic  
- Compare multiple ML models  
- Identify unknown attacks using anomaly detection  
- Implement distributed machine learning using Apache Spark

---

## ML Tasks Covered

- Feature Engineering  
- Feature Selection  
- Classification  
- Anomaly Detection  
- Model Evaluation  

---

## Tech Stack

- **Frontend:** Streamlit  
- **Backend:** Python  
- **Distributed Computing:** Apache Spark 
- **ML Libraries:** Scikit-learn, SynapseML  
- **Visualization:** Matplotlib, Seaborn  

---

## Dataset

- **Dataset Used:** NSL-KDD  
- Contains multiple attack types (e.g., neptune, smurf, etc.)

### Data Handling

- Original labels → used for visualization  
- Converted to binary:
  - `0 → Normal`
  - `1 → Attack`

---

## Pipeline Workflow

1. Upload Dataset  
2. Data Cleaning  
3. Feature Engineering  
4. Model Training (Distributed Spark)  
5. Evaluation & Results  

**Flow Representation:**

Data → Cleaning → Feature Engineering → Model Training → Evaluation

---

## Models Implemented

| Model | Type | Role in System | Key Insight |
|------|------|--------------|------------|
| Random Forest | Supervised | Primary classifier | Achieves highest accuracy (~98%) |
| Logistic Regression | Supervised | Baseline model | Fast and interpretable (~91%) |
| Isolation Forest | Unsupervised | Anomaly detection | Detects unknown/zero-day attacks |
| Hybrid RF + IF | Combined | Robust detection | Uses RF + anomaly override |
| Hybrid LR + IF | Combined | Lightweight hybrid | Combines speed + anomaly detection |

---

## Visualizations

- Bar Chart → All attack types  
- Pie Chart → Normal vs Attack  
- Heatmap → Feature correlation  
- Model comparison graphs  
- Confusion Matrix  

---

## Project Structure

```
DS/
│── app.py                    # Streamlit UI
│── model_trainer.py          # Distributed Spark training
│── feature_extractor.py      # Feature engineering pipeline
│── install.sh                # Setup script
│── start.sh                  # Run script
│── requirements              # Dependencies

````

---

## Distributed Spark Setup

This project uses **Apache Spark in distributed mode** for scalable data processing and model training.

The system follows a **master–worker architecture**, where:
- The **Master node** manages resource allocation and task scheduling  
- The **Worker nodes** execute distributed computations in parallel  

This setup enables efficient handling of large-scale network data and improves overall performance.

---

## Step 1: Enable SSH

```bash
sudo apt update
sudo apt install openssh-server
sudo service ssh start
````

---

## Step 2: Setup Passwordless SSH

```bash
ssh-keygen -t rsa -P ""
ssh-copy-id localhost
```
(Repeat for all worker IPs)

---

## Step 3: Configure Spark

Edit:
```bash
nano $SPARK_HOME/conf/spark-env.sh
```

Add:
```bash
export SPARK_MASTER_HOST=localhost
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
```

---

## Step 4: Configure Workers

Edit:
```bash
nano $SPARK_HOME/conf/workers
```

Add worker nodes in this format:
```
worker_username@<worker_ip>
```
> Ensure all worker nodes are accessible via SSH for proper distributed execution.

---

## Step 5: Start Spark Cluster

```bash
cd spark
sbin/start-all.sh
cd ..
```

---

## Step 6: Setup Virtual Environment

```bash
cd network_ids
source venv/bin/Activate.ps1
```


---

## Step 7: Run Application

```bash
./start.sh
```

---

## Step 8: Open in Browser

```
http://localhost:8501
```

---

## How Distributed Processing Works

- Spark splits data into partitions  
- Each worker processes a subset of the dataset in parallel  
- Workers compute partial results during model training  
- These results are aggregated to form a single global model  
- This improves scalability and performance  

---

## Results

- Random Forest achieved the highest accuracy (~98%), outperforming other models  
- Logistic Regression provided a fast baseline with good performance (~91%)  
- Hybrid models (RF + IF, LR + IF) enhance robustness by combining classification with anomaly detection  
- Isolation Forest helps identify anomalous or previously unseen attack patterns, though with lower accuracy  

---

## Key Features

* End-to-end ML pipeline
* Distributed model training
* Hybrid detection system
* Interactive UI
* Real-world cybersecurity application

---

## Future Improvements

- Real-time intrusion detection using streaming data (e.g., Spark Streaming or Kafka)  
- Containerization and deployment using Docker  
- Integration of deep learning models (LSTM, Autoencoders)  
- Deployment with live network traffic monitoring systems  

---

## Contributors

* Sanjana Kozhipurath
* Priyanshi Vikram Mehta
* Bodhini Jain
* Kavya Fagun Shah
