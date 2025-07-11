from django.shortcuts import render, redirect,get_object_or_404 ## ure actual code logi in this page
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.models import User
import cv2
from django.contrib import messages
import mediapipe as mp
import numpy as np
import pickle
from .models import newhrvdb, Uploaddataset,ModelAccuracy,SystemLog,ModelAccuracy,Prediction,DatasetUsage
import csv
from django.http import HttpResponse
from .forms import Datasetuploadform
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import os
from django.db.models import Avg,Count
from django.http import JsonResponse
from django.utils.timezone import now
from django.db.models.functions import TruncMonth
from django.db.models import Count
import json




model_path = r'C:\Users\gunda\OneDrive\Desktop\major_project\majorproject\pupilheart\static\models\model.keras'
model = load_model(model_path)
    

scaler_path = r'C:\Users\gunda\OneDrive\Desktop\major_project\majorproject\pupilheart\static\models\scaler.pkl'
with open(scaler_path,'rb') as file:
    scaler = pickle.load(file)

def register_page(request):
    if request.method == 'POST':
        username = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("password")
        confirm_password = request.POST.get("confirm_password")
        first_name = request.POST.get("first_name")
        last_name = request.POST.get("last_name")

        if password != confirm_password:
            return render(request, "registration.html", {"error": "Password and confirm password mismatch"})

        user = User.objects.create_user(
            username=username,
            password=password,
            email=email,
            first_name=first_name,
            last_name=last_name
        )

        return redirect("login")

    return render(request, "registration.html")



def login_page(request):
    if request.method == 'POST':
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request,username=username,password=password)
        if user is not None:
            login(request,user)
            return redirect("dashboard")
        else:
            return render(request, "login.html", {"error": "Invalid credentials"})
        
    return render(request,"login.html")

    
def home_page(request):
    return render(request,"homepage.html")

def logout_page(request):
    logout(request)
    return redirect ("login")

def dashboard(request):
    return render(request,"dashboard.html")

def find_hrv(request):
    pupil_sizes = []

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    cap = cv2.VideoCapture(0)
    frame_count = 0
    TARGET_FRAMES = 60

    while cap.isOpened() and frame_count < TARGET_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_frame)

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                left_eye_indices = [474, 475, 476, 477]
                right_eye_indices = [469, 470, 471, 472]

                left_points = [(int(face_landmarks.landmark[i].x * w),
                                int(face_landmarks.landmark[i].y * h)) for i in left_eye_indices]
                right_points = [(int(face_landmarks.landmark[i].x * w),
                                 int(face_landmarks.landmark[i].y * h)) for i in right_eye_indices]

                def euclidean(p1, p2):
                    return np.linalg.norm(np.array(p1) - np.array(p2))

                left_diameter = euclidean(left_points[0], left_points[2])
                right_diameter = euclidean(right_points[0], right_points[2])
                avg_pupil_size = (left_diameter + right_diameter) / 2.0

                pupil_sizes.append(round(avg_pupil_size, 2))
                frame_count += 1

        cv2.imshow("Pupil Tracker", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # If 60 values captured, predict HRV
    if len(pupil_sizes) == 60:
        input_array = np.array(pupil_sizes).reshape(1, -1)
        scaled = scaler.transform(input_array).reshape(1, 60, 1)
        prediction = model.predict(scaled)
        predicted_hrv = round(prediction[0][0], 2)

        # Save to database
        hrv_entry = newhrvdb(hrv_prediction=predicted_hrv,user=request.user)
        for i, value in enumerate(pupil_sizes, start=1):
            setattr(hrv_entry, f'pupil_{i}', float(value))
        hrv_entry.save()

    else:
        predicted_hrv = "Insufficient data"

    return render(request, "dashboard.html", {"pupil_sizes": pupil_sizes, "predicted_hrv": predicted_hrv})

def viewresults(request):
    if not request.user.is_authenticated:
        return redirect("login")
    predictions = newhrvdb.objects.filter(user=request.user).order_by('-id')[:5]
    return render(request,"view_predictions.html",{"predictions":predictions})



def downloadresults(request):

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="hrv_predictions.csv"'

    writer = csv.writer(response)
    
    
    header = ['ID', 'HRV Prediction', 'HRV Detection'] + [f'Pupil_{i}' for i in range(1, 61)]
    writer.writerow(header)

   
    for entry in newhrvdb.objects.all():
        row = [entry.id, entry.hrv_prediction, entry.hrv_detection_type]
        for i in range(1, 61):
            row.append(getattr(entry, f'pupil_{i}'))
        writer.writerow(row)

    return response


@login_required
def user_prediction_chart(request):
    # Count predictions by HRV type (Low, Normal, High)
    chart_data = (
        newhrvdb.objects
        .filter(user=request.user)
        .values('hrv_detection_type')
        .annotate(count=Count('id'))
    )

    labels = [entry['hrv_detection_type'] for entry in chart_data]
    data = [entry['count'] for entry in chart_data]

    return render(request, 'user_prediction_chart.html', {
        'labels': labels,
        'data': data,
    })


## test function
def is_superuser(user):   ## allow only admins
    return user.is_superuser


## creating admin login

def adminlogin(request):
    if request.method == 'POST':
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            if user.is_superuser:
                return redirect("admindashboard")  
            else:
                return redirect("dashboard") 
        else:
            return render(request, "adminlogin.html", {"error": "Invalid credentials"})
    return render(request, "adminlogin.html")


def admindashboard(request):
    total_datasets = Uploaddataset.objects.count()
    total_users = User.objects.count() - 1  # maybe exclude admin
    total_logs = SystemLog.objects.count()
    total_predictions = Prediction.objects.count()

    # Fetch accuracy records (same as accuracyview)
    accuracy_entries = ModelAccuracy.objects.order_by('-timestamp')[:10]
    labels = [entry.timestamp.strftime("%Y-%m-%d %H:%M") for entry in accuracy_entries[::-1]]
    data = [round(entry.accuracy * 100, 2) for entry in accuracy_entries[::-1]]  # Multiply by 100 for %
    
    context = {
        'total_datasets': total_datasets,
        'total_users': total_users,
        'total_logs': total_logs,
        'total_predictions': total_predictions,
        'accuracy_chart_data': {
            'labels': json.dumps(labels),
            'data': json.dumps(data)
        }
    }
    return render(request, 'admindashboard.html', context)




@login_required ## it make sures only loginedin users can raech to manageuusers.html page otherwise django redirect its to login page
@user_passes_test(is_superuser)
def manageusers(request):
    users = User.objects.exclude(id=request.user.id) ## excluding the current user
    return render(request,"manageusers.html",{"users":users})

@login_required
@user_passes_test(is_superuser) # Runs a test function (is_superuser) on the logged-in user. Only users passing this test (i.e., admins) can run this view. Others get a 403 Forbidden or are redirected.

def activateuser(request,userid):
    user = get_object_or_404(User,id=userid)
    user.is_active = True
    user.save()
    return redirect("manageuser")

@login_required
@user_passes_test(is_superuser)

def deactivateuser(request,userid):
    user = get_object_or_404(User,id=userid)
    user.is_active = False
    user.save()
    return redirect("manageuser")

@login_required
@user_passes_test(is_superuser)

def deleteusers(request,userid):
    user = get_object_or_404(User,id=userid)
    user.delete()
    return redirect("manageuser")


def uploaddataset(request):
    if request.method == 'POST':
        form = Datasetuploadform(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('retrainmodel')  
    else:
        form = Datasetuploadform()  

    return render(request, 'uploaddataset.html', {'form': form})


@login_required
@user_passes_test(is_superuser)



def retrainmodel(request):
    try:
        global model, scaler  # Ensure global model and scaler can be reused

        datasets = Uploaddataset.objects.all()
        if not datasets:
            messages.error(request, 'No datasets found for retraining.')
            return redirect('uploaddataset')

        combined_df = pd.DataFrame()

        # Required column names
        required_columns = ['ID', 'HRV Prediction', 'HRV Detection'] + [f'Pupil_{i+1}' for i in range(60)]

        for dataset in datasets:
            try:
                df = pd.read_csv(dataset.file.path)

                # Normalize and rename columns
                df.columns = df.columns.str.strip()
                df.rename(columns={
                    'Id': 'ID',
                    'HRV_value': 'HRV Prediction',
                    'HRV_class': 'HRV Detection',
                    **{f'Pupil{i+1}': f'Pupil_{i+1}' for i in range(60)}
                }, inplace=True)

                # Debugging column output (optional)
                print(f"Columns in {dataset.name}: {df.columns.tolist()}")

                # Check for missing columns
                if not all(col in df.columns for col in required_columns):
                    missing = [col for col in required_columns if col not in df.columns]
                    messages.error(request, f'Dataset "{dataset.name}" missing columns: {missing}')
                    return redirect('uploaddataset')

                combined_df = pd.concat([combined_df, df[required_columns]], ignore_index=True)

            except Exception as inner_e:
                messages.error(request, f'Error reading {dataset.name}: {str(inner_e)}')
                return redirect('uploaddataset')

        # Extract features and labels
        X = combined_df[[f'Pupil_{i+1}' for i in range(60)]].values
        y = combined_df['HRV Prediction'].values

        # Scale and reshape
        X_scaled = scaler.fit_transform(X)
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], 60, 1))

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

        # Retrain the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

        # Evaluate
        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        accuracy = 1 - mae
        ModelAccuracy.objects.create(accuracy=accuracy)
        messages.success(request, f'Model retrained successfully! Test MAE: {mae:.2f}')

        # Save updated scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        return redirect('uploaddataset')

    except Exception as e:
        messages.error(request, f'Error retraining model: {str(e)}')
        return redirect('uploaddataset')
        

def viewdatasets(request):
    try:
        datasets = Uploaddataset.objects.all().order_by('-datetime')
        if not datasets.exists():
            messages.warning(request,"No datasets have been uploaded yet")
        return render(request,'viewdatasets.html',{'datasets':datasets})
    
    except Exception as e:
        messages.warning(request,f'Error retrieving datasets :{str(e)}')
        return render(request,'viewdatasets.html',{'datasets':[]})
    
@login_required
@user_passes_test(is_superuser)
def deletedataset(request, dataset_id):
    try:
        dataset = get_object_or_404(Uploaddataset, id=dataset_id)
        dataset_name = dataset.name
        file_path = dataset.file.path
        dataset.delete()
        if os.path.exists(file_path):
            os.remove(file_path)
        messages.success(request, f"Dataset '{dataset_name}' deleted successfully.")
    except Exception as e:
        messages.error(request, f"Error deleting dataset: {str(e)}")
    return redirect('viewdatasets')


def accuracyview(request):
    # Fetch last 10 accuracy records (or all if you want)
    accuracy_entries = ModelAccuracy.objects.order_by('-timestamp')[:10]

    # Prepare data for Chart.js
    labels = [entry.timestamp.strftime("%Y-%m-%d %H:%M") for entry in accuracy_entries[::-1]]
    data = [entry.accuracy for entry in accuracy_entries[::-1]]

    context = {
        "accuracy_chart_data": {
            "labels": labels,
            "data": data
        }
    }
    return render(request, 'modelaccuracy.html', context)


@login_required
def system_logs_view(request):
    logs = SystemLog.objects.select_related('user').order_by('-timestamp')[:50]
    return render(request, 'system_logs.html', {'logs': logs})

def logout_view(request):
    logout(request)
    return redirect('adminlogin')


def some_view(request):
    if request.user.is_authenticated:
        SystemLog.objects.create(
            user=request.user,
            action='Visited some view',
            timestamp=now()
        )




   