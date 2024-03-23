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
            image_arr = np.array(image).reshape(1, 28, 28, 1).astype('float32') / 255.0  # Reshape to 28x28x1
            payload = json.dumps({"instances": image_arr.tolist()})

            # URL of your TensorFlow Serving API
            url = 'http://3.135.64.60:8501/v1/models/webmodel:predict'

            # Headers for the POST request
            headers = {"Content-Type": "application/json"}

            try:
                # Make the POST request
                response = requests.post(url, data=payload, headers=headers)

                predictions_array = np.array(response.json()['predictions'])
                index_of_largest = np.argmax(predictions_array)
                labels = ['airplane', 'alarm clock', 'angel', 'ant', 'apple', 'axe', 'banana', 'bandage', 'barn', 'baseball bat', 'baseball', 'basket', 'basketball', 'bat', 'bee', 'bicycle', 'bird', 'bridge', 'butterfly', 'calculator', 'camel', 'camera', 'candle', 'car', 'carrot', 'cat', 'ceiling fan', 'chair', 'clock', 'cloud', 'compass', 'computer', 'cookie', 'crab', 'crown', 'diamond', 'dog', 'door', 'dragon', 'elephant', 'envelope', 'eye', 'fire hydrant', 'fish', 'flower', 'fork', 'giraffe', 'guitar', 'hammer', 'helicopter', 'hot air balloon', 'house', 'key', 'light bulb', 'monkey', 'mountain', 'mouth', 'octopus', 'palm tree', 'paper clip', 'pencil', 'piano', 'pineapple', 'rhinoceros', 'sandwich', 'scissors', 'shark', 'sheep', 'shoe', 'shorts', 'skateboard', 'skull', 'smiley face', 'snail', 'snowflake', 'snowman', 'soccer ball', 'sock', 'spider', 'star', 'stop sign', 'sun', 'sword', 't-shirt', 'television', 'tennis racquet', 'The Eiffel Tower', 'The Great Wall of China', 'The Mona Lisa', 'tooth', 'toothbrush', 'train', 'tree', 'windmill']
                prediction = labels[index_of_largest]
            except requests.exceptions.RequestException as e:
                # Handle any exceptions that arise during the request
                prediction = 'Error making request'

            # Send the prediction back to the frontend
            await self.send(text_data=json.dumps({'prediction': str(prediction)}))
