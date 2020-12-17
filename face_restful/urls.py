from django.urls import path
from . import views

app_name = 'face'

urlpatterns = [
    path('recognition/', views.face_recognition, name='recognition'),
    path('signin/', views.signin, name='signin')

]
