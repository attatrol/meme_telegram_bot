FROM tensorflow/tensorflow:1.15.4-gpu-py3
CMD nvidia-smi

RUN pip install pillow
RUN pip install gpt-2-simple
RUN pip install pandas
RUN pip install python-telegram-bot --upgrade

WORKDIR /usr/local/
COPY src ./
CMD [ "python", "./bot.py" ]
