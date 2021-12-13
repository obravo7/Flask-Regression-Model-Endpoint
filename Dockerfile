FROM python:3.7-slim

# set work dir
COPY . /priceapi/app
WORKDIR /priceapi/app

# install nginx
RUN apt-get clean \
    && apt-get -y update

RUN apt-get -y install nginx \
    && apt-get -y install python3-dev \
    && apt-get -y install build-essential

# install requirements
RUN pip install -r requirements.txt

# EXPOSE 5000
# ENV FLASK_APP=app.py

# move nginx config to proper location
COPY nginx.conf /etc/nginx
RUN chmod +x ./start.sh
CMD ["./start.sh"]
