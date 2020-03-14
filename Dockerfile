FROM continuumio/anaconda
LABEL maintainer="ronniesong0809@gmail.com"
EXPOSE 5000
COPY . /app
WORKDIR /app
RUN conda install -c conda-forge poppler
RUN pip install -r requirements.txt
CMD ["python", "app.py"]