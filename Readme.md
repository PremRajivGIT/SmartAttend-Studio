

# Yo! How's your Day Going! 😎

## Anyways, refer to these instructions to run this website/server/studio :)



### 1. Install WSL (Important!)

Follow these steps:

```bash
wsl --install
```

### 2. Clone this repository (hint: use git clone in WSL(Ubuntu))

### 3. Install requirements.txt

(TIP: Create a venv or rewrite source packages 💀 xd)
***Note: Use python3.11 for better compatibility***

```bash
# install python 3.11
sudo apt update && sudo apt upgrade -y
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.11 python3.11-venv -y

# create venv
python3.11 -m venv venv
source venv/bin/activate

# install requirements
pip install --upgrade pip
pip install -r requirements.txt
```



### 4. You need to Download these files and move them to the project directory (tbc)

* Face detection model: [`yolov8m_200e.pt`](https://drive.google.com/file/d/1IJZBcyMHGhzAi0G4aZLcqryqZSjPsps-/view?usp=sharing) → put inside `The Root Directory`
* Credits for the Yolo Face Model, Visit -> [https://github.com/Yusepp/YOLOv8-Face](https://github.com/Yusepp/YOLOv8-Face) 

---

### 5. Run app.py

```bash
python3.11 app.py
```

Visit 👉 [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.



# 📚 Project Info

## 🚀 Features

* 📷 **YOLOv8-based face detection** (`yolov8m_200e.pt`)
* 🧠 **Face embeddings** with TensorFlow/Keras FaceNet
* 🔄 **Real-time attendance marking**
* 🏫 **Multi-classroom support**
* 🗄 **Database integration** with SQLite
* 📊 **Model training & export** (ONNX / TFLite)



## 🖥️ How to Use the Model

* You Need Our Mobile Application to Run the exported model -> [To Be Updated](https://images.alphacoders.com/137/1377812.png)
* Git Repo for our Mobile Application -> [Smart Attend Mobile APP](https://github.com/CH-V-N-Rugvidh/SmartAttendMobile)





## 📁 Project Structure

```
SmartAttend-Studio/
│── app.py                 # Flask entrypoint
│── requirements.txt       # Python dependencies
│──  yolov8m_200e.pt       # YOLOv8 face detection model
│── static/                # Frontend assets
│── templates/             # HTML templates
│── utils/                 # Helper scripts (training, inference)
│── database/              # SQLAlchemy models & migrations
```



