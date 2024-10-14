# Use the official Jupyter minimal notebook image
FROM jupyter/minimal-notebook:latest

# Set the working directory in the container
WORKDIR /home/jovyan/work

# Copy the local notebook file and other files to the container
COPY notebook.ipynb /home/jovyan/work/notebook.ipynb
COPY requirements.txt /home/jovyan/work/requirements.txt
COPY dataset /home/jovyan/work/dataset
COPY nets /home/jovyan/work/nets
COPY resources /home/jovyan/work/resources
COPY utils /home/jovyan/work/utils

# Switch to root user to install system packages
USER root

# Install Python venv and any other needed packages
RUN apt-get update && apt-get install -y python3-venv

# Create a Python virtual environment inside the container
RUN python3 -m venv /opt/venv

# Activate the virtual environment and install dependencies
RUN /bin/bash -c "source /opt/venv/bin/activate && pip install --no-cache-dir -r requirements.txt"

# Ensure the container uses the virtual environment for Python
ENV PATH="/opt/venv/bin:$PATH"

# Expose the port Jupyter Notebook will run on
EXPOSE 8888

# Start Jupyter Notebook in the virtual environment
CMD ["bash", "-c", "source /opt/venv/bin/activate && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"]
