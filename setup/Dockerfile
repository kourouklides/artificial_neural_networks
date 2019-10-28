FROM continuumio/miniconda3

# Set the ENTRYPOINT to use bash
ENTRYPOINT [ "/bin/bash", "-c" ]

# Use the environment.yml to create the conda environment
ADD environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml

# Pull the environment name out of the environment.yml
RUN echo "source activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH

#TODO: complete the rest of Dockerfile
