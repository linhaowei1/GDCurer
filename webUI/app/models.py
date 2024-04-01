from django.contrib.auth.models import User
from django.db import models
from django.dispatch import receiver
from django.db.models.signals import post_save


# Create our own DBUser based on django.contrib.auth.models.User
# According to https://docs.djangoproject.com/zh-hans/4.0/topics/auth/customizing/#extending-the-existing-user-models

class Doctor(models.Model):
    user = models.OneToOneField(
        User, on_delete=models.CASCADE, related_name='doctor')
    name = models.CharField("姓名", max_length=10, null=True)

    class Gender(models.IntegerChoices):
        MALE = (0, "男")
        FEMALE = (1, "女")

    gender = models.SmallIntegerField(
        "性别", choices=Gender.choices, null=True, blank=True
    )

    def __str__(self):
        return str(self.name)


class Patient(models.Model):

    pid = models.CharField("ID", max_length=100, null=True)
    age = models.SmallIntegerField("Age", null=True, blank=True)
    
    class Gender(models.IntegerChoices):
        MALE = (0, "男")
        FEMALE = (1, "女")

    gender = models.SmallIntegerField(
        "性别", choices=Gender.choices, null=True, blank=True
    )
    
    def __str__(self):
        return str(self.ID)


class Prediction(models.Model):
    class Meta:
        indexes = [models.Index(fields=['pid', 'poster'])]

    pid = models.AutoField(primary_key=True)
    poster = models.ForeignKey(Doctor, on_delete=models.CASCADE)
    patient_id = models.ForeignKey(Patient, on_delete=models.CASCADE)
    half_life = models.FloatField("Half life of I131 /day")
    max_uptake = models.FloatField("Maximum I-131 Uptake /h")
    dicom_img = models.ImageField('Dicom image', upload_to=f'dicom/')
    predicted_dosage = models.FloatField("Maximum I-131 Uptake /h", default=0)
    created_time = models.DateTimeField('创建时间', auto_now_add=True)


class TrainData(models.Model):
    class Meta:
        indexes = [models.Index(fields=['pid', 'poster'])]

    pid = models.AutoField(primary_key=True)
    poster = models.ForeignKey(Doctor, on_delete=models.CASCADE)
    # Patient related,
    patient_id = models.ForeignKey(Patient, on_delete=models.CASCADE)
    half_life = models.FloatField("Half life of I131 /day")
    max_uptake = models.FloatField("Maximum I-131 Uptake /h")
    dicom_img = models.ImageField('Dicom image', upload_to=f'train_dicom/')
    # predicted_dosage = models.FloatField("Maximum I-131 Uptake /h", default=0)

    class Label(models.IntegerChoices):
        NORMAL = (0, "Normal")
        HYPO = (1, "Hypothyroidsm")
        HYPER = (2, "Hyperthyroidsm")

    label = models.SmallIntegerField("Label", choices=Label.choices)

    created_time = models.DateTimeField('创建时间', auto_now_add=True)


class Performance(models.Model):

    phase_time = models.DateTimeField('phase time')
    pearson_correlation = models.FloatField("Pearson Correlation")
    weighted_precision = models.FloatField("Weighted Precision")
    hyper_precision = models.FloatField("Precision for Hyperthyroidism Cases")
    hypo_precision = models.FloatField(
        "Precision for Hypothyroidism Cases")
    model_addr = models.CharField("Model Address", max_length=100)


