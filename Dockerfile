FROM loganbickmore/face_recognition:latest

WORKDIR /app

COPY . .

ENTRYPOINT

CMD ["python3", "main.py"]
