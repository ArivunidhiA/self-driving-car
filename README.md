# 🚗 Advanced Self-Driving Car Project: End-to-End Deep Learning Pipeline

This project builds an end-to-end deep learning pipeline that trains a self-driving car model using computer vision and deploys it in real-time on the Udacity simulator. It was developed entirely from scratch by collecting custom driving data, preprocessing it, training a convolutional neural network (CNN), and deploying the model using a real-time server.

---

## 📽️ Demo

<table>
<tr>
<td>Before Preprocessing</td>
<td>After Preprocessing</td>
</tr>
<tr>
<td><img src="examples/before_preprocessing.jpg" width="350"/></td>
<td><img src="examples/after_preprocessing.jpg" width="350"/></td>
</tr>
</table>

🎥 [Watch a demo clip of the car driving autonomously](examples/driving_clip.mp4)

---

## 📂 Project Structure

```
self-driving-car-project/
├── etl_pipeline.ipynb        # Jupyter notebook for data cleaning, preprocessing, and training
├── drive.py                  # Real-time inference server using Flask and SocketIO
├── model.h5                  # Trained CNN model (or provide Google Drive link)
├── requirements.txt          # Python dependencies
├── examples/                 # Screenshots & demo video
│   ├── before_preprocessing.jpg
│   ├── after_preprocessing.jpg
│   └── driving_clip.mp4
├── raw_data/ (optional)      # Sample dataset (csv + few images)
│   ├── driving_log.csv
│   └── IMG/
└── README.md                 # This file
```

---

## 🧠 Project Overview

### 🎯 Goal:
Train a deep learning model to predict steering angles from road images and deploy it to drive a virtual car in real time.

### 🔁 Workflow:
1. **Data Collection** — Drove in the Udacity simulator to collect images & steering angles
2. **Cloud Storage** — Uploaded dataset to Amazon S3
3. **ETL Pipeline** — Cleaned, preprocessed, and augmented data using Python
4. **Model Training** — Trained a CNN using Keras/TensorFlow
5. **Real-Time Deployment** — Used `drive.py` to send predictions back to the simulator

---

## 🧪 Data Preprocessing

- Cropped image (remove sky and hood)
- Resized to `(200, 66)` for NVIDIA model
- Converted to YUV color space
- Normalized pixel values
- Augmentation:
  - Brightness adjustment
  - Horizontal flips
  - Random translations (simulate lane shifts)

---

## 🏗️ Model Architecture (Inspired by NVIDIA)

```python
model = Sequential()
model.add(Lambda(lambda x: x, input_shape=(66, 200, 3)))
model.add(Conv2D(24, (5,5), strides=(2,2), activation='elu'))
model.add(Conv2D(36, (5,5), strides=(2,2), activation='elu'))
model.add(Conv2D(48, (5,5), strides=(2,2), activation='elu'))
model.add(Conv2D(64, (3,3), activation='elu'))
model.add(Conv2D(64, (3,3), activation='elu'))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
```

- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Adam (1e-4)
- **Trained:** 5 epochs on ~17,000 images

---

## 🧰 Tech Stack

- Python
- TensorFlow / Keras
- OpenCV / NumPy / Matplotlib
- Flask + SocketIO
- Amazon S3 (for storage)
- Jupyter Notebook
- Udacity Self-Driving Car Simulator

---

## ▶️ How to Run

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/self-driving-car-project.git
cd self-driving-car-project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Flask Server

```bash
python drive.py
```

### 4. Launch Udacity Simulator
- Select "Autonomous Mode"
- Car will begin driving using your model!

---

## 📝 Acknowledgements
This project is inspired by Udacity's Self-Driving Car Nanodegree and NVIDIA's end-to-end learning paper.

---

## 📬 Contact

Built by **Arivunidhi Anna Arivan**  
📧 [Your email]  
🔗 [LinkedIn](https://www.linkedin.com/in/yourprofile)  
🌐 [Portfolio (if any)]

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
