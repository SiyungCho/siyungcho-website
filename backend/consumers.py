import json
from io import BytesIO

from channels.generic.websocket import AsyncWebsocketConsumer
import numpy as np
from PIL import Image
import requests

class BackendConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data=None, bytes_data=None):
        if bytes_data:
            image = Image.open(BytesIO(bytes_data)).convert('L')  # Convert to grayscale

            image = image.resize((28, 28), Image.Resampling.LANCZOS)
            img_array = np.array(image)
            img_array = img_array.reshape(1, 28, 28, 1).astype('float32') / 255.0
        
            payload = json.dumps({"instances": img_array.tolist()})

            # URL of your TensorFlow Serving API
            url = 'http://3.142.235.33:8501/v1/models/webmodel:predict'

            # Headers for the POST request
            headers = {"Content-Type": "application/json"}

            try:
                # Make the POST request
                response = requests.post(url, data=payload, headers=headers)
                print(response.json())

                predictions_array = np.array(response.json()['predictions'])
                index_of_largest = np.argmax(predictions_array)
                labels = ['apple', 'banana', 'bicycle', 'butterfly', 'camel', 'camera', 'car', 'clock', 'cloud', 'cookie', 'guitar', 'house', 'light bulb', 'octopus', 'paper clip', 'scissors', 'shoe', 'shorts', 'snowman', 'star', 'stop sign', 'sun', 't-shirt', 'television', 'tennis racquet', 'tree']
                prediction = labels[index_of_largest]
                print(prediction)
            except requests.exceptions.RequestException as e:
                # Handle any exceptions that arise during the request
                prediction = 'Error making request'

            # Send the prediction back to the frontend
            await self.send(text_data=json.dumps({'prediction': str(prediction)}))
