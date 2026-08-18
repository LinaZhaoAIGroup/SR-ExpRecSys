from django.urls import path
from . import views

app_name = 'chat'

urlpatterns = [
    path('', views.index, name='index'),
    path('api/conversations/', views.get_conversations, name='get_conversations'),
    path('api/conversations/create/', views.create_conversation, name='create_conversation'),
    path('api/conversations/<int:conversation_id>/', views.get_messages, name='get_messages'),
    path('api/conversations/<int:conversation_id>/delete/', views.delete_conversation, name='delete_conversation'),
    path('api/conversations/<int:conversation_id>/title/', views.update_conversation_title, name='update_conversation_title'),
    path('api/messages/send/', views.send_message, name='send_message'),
]

