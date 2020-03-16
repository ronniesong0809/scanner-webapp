FROM ubuntu:18.04
MAINTAINER Your Name "ronniesong0809@gmail.com"
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev
RUN apt-get install -y poppler-utils libsm6 libxext6
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python3"]
CMD ["app.py"]