from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/backend/$', consumers.BackendConsumer.as_asgi()),
]