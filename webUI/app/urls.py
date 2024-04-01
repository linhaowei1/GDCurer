from django.urls import path

from . import views

app_name = 'app'
urlpatterns = [
    path('', views.InferenceView.as_view(), name='index_view'),
    path('login/', views.LoginView.as_view(), name='login_view'),
    path("logout/", views.LogoutView.as_view(), name="logout_view"),
    path('training/', views.TrainView.as_view(), name='train_view'),
    path('performance/', views.PerformanceView.as_view(), name='performance_view'),
    path('addtrain/', views.AddTrainView.as_view(), name='addtrain'),
]