FROM python:3.10.11

COPY requirements.txt .

RUN apt-get -y update
RUN apt-get install ffmpeg
RUN python -m pip install --no-cache-dei -r requirements.txt

ADD ./data /data
ADD src/box-markup.py /

CMD ["python", "./box-markup.py"]
