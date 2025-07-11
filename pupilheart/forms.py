from django import forms
from .models import Uploaddataset


class Datasetuploadform(forms.ModelForm):
    class Meta:
        model = Uploaddataset
        fields = ['name','file']
        widgets = {   ## to match the form styling with css styling
            'name': forms.TextInput(attrs={'class': 'styled-input'}),
            'file': forms.ClearableFileInput(attrs={'class': 'styled-input'}),
        }