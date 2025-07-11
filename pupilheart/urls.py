from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path("",views.home_page,name="home"),
    path("register/",views.register_page,name="register"),
    path("login/",views.login_page,name="login"),
    path("logout/",views.logout_page,name="logout"),
    path("dashboard/",views.dashboard,name="dashboard"),
    path("findhrv/",views.find_hrv,name="findhrv"),
    path("viewresults/",views.viewresults,name="viewresults"),
    path("downloadresults/",views.downloadresults,name="downloadresults"),
    path("adminlogin/",views.adminlogin,name="adminlogin"),
    path("admindashboard/",views.admindashboard,name="admindashboard"),
    path("manageusers/",views.manageusers,name="manageuser"),
    path("activateuser/<int:userid>/", views.activateuser, name="activateuser"),
    path("deactivateuser/<int:userid>/", views.deactivateuser, name="deactivateuser"),
    path("deleteuser/<int:userid>/", views.deleteusers, name="deleteuser"),
    path("uploaddataset/",views.uploaddataset,name="uploaddataset"),
    path("retrainmodel/",views.retrainmodel,name="retrainmodel"),
    path("viewdatasets/", views.viewdatasets, name="viewdatasets"),
    path("deletedatasets/<int:dataset_id>/", views.deletedataset, name="deletedatasets"),
    path('accuracyview/', views.accuracyview, name='accuracyview'),
    path('system-logs/', views.system_logs_view, name='system_logs'),
    path('adminlogout/', views.logout_view, name='adminlogout'),
    path('user/chart/', views.user_prediction_chart, name='user_prediction_chart'),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)