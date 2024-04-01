from app.models import *
from app.utils import get_bardisplay
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db import transaction
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render, redirect
from django.urls import reverse
from django.views.generic import TemplateView

# Base class for all views (except for loginView) to enforce login
class MyView(LoginRequiredMixin, TemplateView):
    login_url = '/login/'


class LoginView(TemplateView):
    template_name = 'app/login.html'

    def post(self, request, *args, **kwargs):
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return HttpResponseRedirect(reverse('app:index_view'))
        else:
            return HttpResponse("Wrong username / password.")


class LogoutView(TemplateView):
    def get(self, request, *args, **kwargs):
        logout(request)
        return HttpResponseRedirect(reverse('app:login_view'))


class InferenceView(MyView):
    def get(self, request, *args, **kwargs):
        person = Doctor.objects.get(user=request.user)
        my_predictions = Prediction.objects.all().filter(
            poster=person
        )
        return render(request, 'app/inference.html', locals())

    def post(self, request, *args, **kwargs):
        person = Doctor.objects.get(user=request.user)
        my_predictions = Prediction.objects.all().filter(
            poster=person
        )
        if 'Inference' in request.POST.keys():
            patient_id = str(request.POST.get('ID'))
            half_life = request.POST.get('half')
            max_uptake = request.POST.get('uptake')
            with transaction.atomic():

                patient, _ = Patient.objects.get_or_create(
                    pid = patient_id
                )
                new_prediction = Prediction.objects.create(
                    poster=person,
                    patient_id=patient,
                    half_life=half_life,
                    max_uptake=max_uptake,
                )
                # TODO: need api to cal dosage
                new_prediction.predicted_dosage = 0.1
                if request.POST.get('img'):
                    new_prediction.dicom_img = request.POST.get('img')
                new_prediction.save()
            return render(request, 'app/inference.html', locals())

        if 'logout' in request.POST:
            logout(request)
            return HttpResponseRedirect(reverse('app:logout_view'))


class TrainView(MyView):
    def get(self, request, *args, **kwargs):
        person = Doctor.objects.get(user=request.user)
        my_datas = TrainData.objects.all().filter(
            poster=person
        )
        for entity in my_datas:
            entity.label = entity.get_label_display()
        train_datas = TrainData.objects.all()
        for entity in train_datas:
            entity.label = entity.get_label_display()
        return render(request, 'app/train.html', locals())

    def post(self, request, *args, **kwargs):
        if 'logout' in request.POST:
            logout(request)
            return HttpResponseRedirect(reverse('app:logout_view'))


class PerformanceView(MyView):
    def get(self, request, *args, **kwargs):
        person = Doctor.objects.get(user=request.user)
        performances = Performance.objects.all().order_by('phase_time')
        pearsonCorrelationData = [p.pearson_correlation for p in performances]
        weightedPrecisionData = [p.weighted_precision for p in performances]
        hyperPrecisionData = [p.hyper_precision for p in performances]
        hypoPrecisionData = [p.hypo_precision for p in performances]
        return render(request, 'app/performance.html', locals())


class AddTrainView(MyView):
    def get(self, request, *args, **kwargs):

        return render(request, 'app/add_train.html', locals())

    def post(self, request, *args, **kwargs):

        person = Doctor.objects.get(user=request.user)

        if 'add_train' in request.POST.keys():
            patient_id = str(request.POST.get('ID'))
            half_life = request.POST.get('half')
            max_uptake = request.POST.get('uptake')
            age = request.POST.get('age')
            label = str(request.POST.get('label'))
            Label2Choice = {"Normal": TrainData.Label.NORMAL,
                           "Hypothyroidsm": TrainData.Label.HYPO, "Hyperthyroidsm": TrainData.Label.HYPER}
            
            with transaction.atomic():
                patient, _ = Patient.objects.get_or_create(
                    pid = patient_id,
                    age=age
                )

                new_train = TrainData.objects.create(
                    poster=person,
                    patient_id=patient,
                    half_life=half_life,
                    max_uptake=max_uptake,
                    label=Label2Choice[label],
                )
                if request.FILES.get('img'):
                    new_train.dicom_img = request.FILES.get('img')
                new_train.save()
            return HttpResponseRedirect(reverse('app:train_view'))
        
        if 'delete' in request.POST.keys():
            pass

        if 'modify' in request.POST.keys():
            pass
        
        if 'logout' in request.POST:
            logout(request)
            return HttpResponseRedirect(reverse('app:logout_view'))
