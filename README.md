
# Simple Machine Learning Application
## Overview
* This project aims to develop a simple machine learning application, containerize it using Docker, and deploy it on GitHub. The application will utilize a pre-trained decision tree model to predict the class of Iris flowers based on input features.


# Step 1: Set Up the VM
Update the System:

```
sudo apt update
sudo apt upgrade -y
```

* Install Necessary Packages:

```
sudo apt install -y curl git
```

# Step 2: Install Docker
Remove Old Versions:
```
sudo apt remove docker docker-engine docker.io containerd runc
```

Set Up the Docker Repository:
```
sudo apt update
sudo apt install -y apt-transport-https ca-certificates
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

Install Docker Engine:
```
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io
```
Verify Docker Installation:
```
sudo docker run hello-world
```
# Step 3: Create a Dockerfile for the ML Application
Create Project Directory:
```
mkdir ml-app
cd ml-app
```
Create a Dockerfile:
Dockerfile
# Use an official Python runtime as a parent image
```FROM python:3.9-slim```

# Set the working directory
```WORKDIR /usr/src/app```

# Copy the current directory contents into the container at /usr/src/app
```COPY . .```

# Install any needed packages specified in requirements.txt
```RUN pip install --no-cache-dir -r requirements.txt```

# Make port 80 available to the world outside this container
```EXPOSE 80```

# Run app.py when the container launches
```
CMD ["python", "app.py"]
Create requirements.txt File:
txt
Copy code
Flask
Numpy
Pandas
scikit-learn
```

# Step 4: Develop the Machine Learning Application
Create a Simple ML Model:
```
python
Copy code
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pickle
```
# Load the Iris dataset
```
iris = load_iris()
X, y = iris.data, iris.target
```
# Train a decision tree classifier
```
clf = DecisionTreeClassifier()
clf.fit(X, y)
```
# Save the model to a file
```
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)
```
Run the Model Training Script:
```
python train_model.py
```
Integrate the Model into the Flask App:
from flask import Flask, request, jsonify
```
import pickle
import numpy as np

app = Flask(__name__)
```
# Load the trained model
```
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def hello_world():
    return 'Hello, Docker!'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict(np.array(data['input']).reshape(1, -1))
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
```
# Step 5: Build and Run the Docker Container
Build the Docker Image:

```
sudo docker build -t ml-app .
```
Run the Docker Container:
```
sudo docker run -p 4000:80 ml-app
```
Access the Application:
Open your browser and navigate to http://localhost:4000 to see the running application.
Test the ML Endpoint:
```
curl -X POST http://localhost:4000/predict -H "Content-Type: application/json" -d '{"input": [5.1, 3.5, 1.4, 0.2]}'
```
# Step 6: Deploy the Application to GitHub
Initialize a Git Repository:
```git init
```
Add All Files and Commit:
```
git add .
git commit -m "Initial commit"
```
Create a New Repository on GitHub:
```
git remote add origin https://github.com/yourusername/your-repository.git
git branch -M main
git push -u origin main
```
