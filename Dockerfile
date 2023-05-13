FROM python:3.8.10

WORKDIR /usr/local/src/sephora-sentiment-analysis

RUN apt-get update && \
  apt-get -y upgrade && \
  apt-get install -y bash 

RUN apt-get install -y libpq-dev python3-dev gcc g++

ENV LANG en_GB.UTF-8
ENV LANGUAGE en_GB:en
ENV LC_ALL en_GB.UTF-8

COPY . ./
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["./run.sh"]
