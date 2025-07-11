from django.db import models   ## code for creating database if it is in ure project
from django.contrib.auth.models import User
from django.utils.timezone import now

"""
class Hrvdb(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    for i in range(1,61):
        locals()[f'pupil_{i}'] = models.FloatField()


   
     hrv_prediction = models.FloatField()

    hrv_detection_type = models.CharField(max_length=20,choices=[
            ('Low', 'Low Variability'),
            ('Normal', 'Normal Variability'),
            ('High', 'High Variability'),
        ],
        default='Low'
    )
"""
class newhrvdb(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    for i in range(1,61):
        locals()[f'pupil_{i}'] = models.FloatField()


    hrv_prediction = models.FloatField()

    hrv_detection_type = models.CharField(max_length=20,choices=[
            ('Low', 'Low Variability'),
            ('Normal', 'Normal Variability'),
            ('High', 'High Variability'),
        ],
        default='Low'
    )


    def save(self, *args, **kwargs):
        if self.hrv_prediction < 40:
            self.hrv_detection_type = 'Low'
        elif 40 <= self.hrv_prediction <= 70:
            self.hrv_detection_type = 'Normal'
        else:
            self.hrv_detection_type = 'High'
        super().save(*args, **kwargs)



class Uploaddataset(models.Model):
    name = models.CharField(max_length=225)
    file = models.FileField(upload_to='datasets/') ## Files are saved in the datasets/ directory relative to the MEDIA_ROOT setting in Django.
    datetime = models.DateTimeField(auto_now_add=True)


    def __str__(self): #Returns the dataset's name as its string representation, useful for display in admin interfaces or querysets.
        return self.name
    

class ModelAccuracy(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    accuracy = models.FloatField()

    def __str__(self):
        return f"{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - Accuracy: {self.accuracy:.2f}"



class SystemLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    action = models.TextField()
    timestamp = models.DateTimeField(default=now)

    def __str__(self):
        return f"{self.user} - {self.action} @ {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"


class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    prediction_type = models.CharField(max_length=100)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.prediction_type}"
    

class DatasetUsage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    dataset_name = models.CharField(max_length=100)
    used_on = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} used {self.dataset_name}"
    

    