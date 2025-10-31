from django.shortcuts import render
from .client import AIClient
import os
from django.conf import settings

def index(request):
    result = ""
    uploaded_file_url = None

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "send_text":
            txt = request.POST.get("txt", "")
            if txt.strip():
                cli = AIClient("Ты честный помощник.")
                result = cli.generate(txt)

        elif action == "upload_file":
            uploaded_file = request.FILES.get("file")
            if uploaded_file:
                file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)
                os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
                with open(file_path, "wb") as f:
                    for chunk in uploaded_file.chunks():
                        f.write(chunk)
                uploaded_file_url = settings.MEDIA_URL + uploaded_file.name

    return render(request, "index.html", {"res": result, "file_url": uploaded_file_url})
