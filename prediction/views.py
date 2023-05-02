import joblib
import numpy as np
from django.shortcuts import render


def options():
    STUDENT_TYPE = {"Generic": 1, "Matured": 2}
    EMPLOYMENT = {"Self Employed": 0, "Not Employed": 1, "Employed": 2}
    WITHDRAWAL = {"Yes": 0, "No": 1}
    REPEAT = {"Yes": 0, "No": 1}
    MARITAL_STATUS = {"Single": 1, "Married": 2}
    GENDER = {"Male": 1, "Female": 2}

    data = {
        'student_types': STUDENT_TYPE.items(),
        'employments': EMPLOYMENT.items(),
        'withdrawals': WITHDRAWAL.items(),
        'repeats': REPEAT.items(),
        'maritals': MARITAL_STATUS.items(),
        'genders': GENDER.items()
    }
    return data


def index(request):
    template = "prediction/index.html"
    context = options()

    return render(request, template, context)


def predict(request):
    template = "prediction/predict.html"
    machine_model = "prediction/model/rfc_model.joblib"
    message = None

    withdrawal = request.POST.get('withdrawal')
    repeat = request.POST.get('repeat')
    marital_status = request.POST.get('marital_status')
    gender = request.POST.get('gender')
    age = request.POST.get('age')
    student_type = request.POST.get('student_type')
    employment = request.POST.get('employment')

    inputs = np.expand_dims([
        int(withdrawal),
        int(repeat),
        int(marital_status),
        int(gender),
        int(student_type),
        int(age),
        int(employment)], 0)

    model = joblib.load(machine_model)
    prediction = model.predict(inputs)
    target = int(prediction)

    if target == 1:
        message = "STUDENT IS LIKELY TO DROPOUT"
    else:
        message = "STUDENT IS NOT LIKELY TO DROPOUT"

    context = {'target': target, 'message': message}

    return render(request, template, context)


