from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User

from .models import *


# admin.site.unregister(User)
admin.site.register(Doctor)
admin.site.register(Prediction)
admin.site.register(TrainData)
admin.site.register(Performance)