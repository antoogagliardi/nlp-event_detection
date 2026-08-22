FROM python:3.9-slim

# setup home directory of the container
WORKDIR /home/app

# install requirements
COPY requirements.txt .
RUN apt-get update && apt-get install --no-install-recommends --yes build-essential
RUN pip install -r requirements.txt

# copy model
COPY model model

# copy code
COPY docker ed
COPY code/src/model.py ed/src/ed_model.py
COPY code/src/utils.py ed/src/utils.py
COPY code/src/plot.py ed/src/plot.py
ENV PYTHONPATH ed

# standard cmd
CMD [ "python", "ed/app.py" ]
