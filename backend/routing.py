from django.urls import path
from . import consumers

websocket_urlpatterns = [
    path('ws/backend/', consumers.BackendConsumer.as_asgi()),
]