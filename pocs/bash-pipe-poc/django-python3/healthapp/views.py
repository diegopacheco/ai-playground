from django.http import JsonResponse, HttpResponse


def health(request):
    return JsonResponse({"status": "ok"})


def root(request):
    return HttpResponse("django-python3", content_type="text/plain")
