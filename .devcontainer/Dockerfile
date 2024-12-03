FROM python:3.11-slim

RUN pip3 install -U pip

COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp


