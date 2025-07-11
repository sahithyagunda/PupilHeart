from django.apps import AppConfig


class PupilheartConfig(AppConfig):
    name = 'pupilheart'
    default_auto_field = 'django.db.models.BigAutoField'

    def ready(self):
        import pupilheart.signals  # ensures the login signal is registered
