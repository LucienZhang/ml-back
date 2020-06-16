FROM python:3.7-slim-buster
WORKDIR /data/project
ENV PYTHONIOENCODING utf-8
ENV LANG en_US.UTF-8

COPY . .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
CMD [ "gunicorn", "ml:app", "-c", "./gunicorn.conf.py" ]
