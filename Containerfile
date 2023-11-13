FROM registry.access.redhat.com/ubi9/python-39:1-143.1697647134

# Add application sources with correct permissions for OpenShift
USER 0

WORKDIR /opt/app-root/src
ADD . /opt/app-root/src/
RUN chown -R 1001:0 /opt/app-root/src/
USER 1001

# Install the dependencies
RUN pip install -r requirements.txt 

# Run the application
CMD uvicorn ols:app --host 0.0.0.0 --port 8080