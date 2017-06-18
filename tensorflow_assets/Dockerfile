FROM luckyfengyong/tensorflow-serving:0.5.1

COPY bootstrap /usr/local/bin/bootstrap
COPY *.py /
# https://github.com/kubernetes-incubator/client-python/blob/master/requirements.txt
#RUN pip install certifi six python-dateutil setuptools urllib3 pyyaml oauth2client ipaddress websocket-client --upgrade
RUN pip install kubernetes --upgrade
