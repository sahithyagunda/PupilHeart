from django.contrib.auth.signals import user_logged_in, user_logged_out
from django.dispatch import receiver
from django.utils.timezone import now
from .models import SystemLog

@receiver(user_logged_in)
def log_user_login(sender, request, user, **kwargs):
    print(f"[SIGNAL FIRED] user_logged_in -> {user.username}")
    SystemLog.objects.create(user=user, action="User logged in", timestamp=now())

@receiver(user_logged_out)
def log_user_logout(sender, request, user=None, **kwargs):
    username = user.username if user else getattr(request, 'user', None).username
    print(f"[SIGNAL FIRED] user_logged_out -> {username}")
    SystemLog.objects.create(user=user, action="User logged out", timestamp=now())
