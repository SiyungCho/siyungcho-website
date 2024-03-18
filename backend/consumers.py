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
            # Convert the image to the required format for the model
            image = Image.open(BytesIO(bytes_data)).convert('L')  # Convert to grayscale
            image_arr = np.array(image).reshape(1, 28, 28, 1)  # Reshape to 28x28x1

            # Prepare the payload for TensorFlow Serving
            data = json.dumps({"instances": image_arr.tolist()})
            headers = {"content-type": "application/json"}
            tf_serving_url = 'http://localhost:8501/v1/models/your_model:predict'
            
            # Make the request to TensorFlow Serving
            response = requests.post(tf_serving_url, data=data, headers=headers)
            prediction = response.json()

            # Send the prediction back to the frontend
            await self.send(text_data=json.dumps({'prediction': str(prediction)}))
