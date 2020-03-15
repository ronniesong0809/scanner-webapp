FROM ubuntu:18.04
MAINTAINER Your Name "ronniesong0809@gmail.com"
RUN apt-get update -y
RUN apt-get install -y python-pip
RUN apt-get install -y poppler-utils libsm6 libxext6
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]