FROM python:3.8.5

RUN mkdir src
WORKDIR src/

COPY . .

RUN python -m pip install --upgrade pip
RUN pip install datawig
RUN pip install openml
RUN pip install pyod
RUN pip install numpy
RUN pip install matplotlib
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install tensorflow
RUN pip install keras
RUN pip install openml
RUN pip install pyod
RUN pip install mxnet
RUN pip install autogluon
RUN pip install mxnet-mkl --pre


CMD [ "python3", "experimental_setup_final.py" ]
