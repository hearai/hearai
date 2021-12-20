FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu18.04

RUN apt-get -y update --fix-missing && apt-get -y upgrade
RUN apt-get install -y sudo wget bzip2 git vim cmake unzip mc nano

# settings
ARG user_name=ubuntu
ARG user_id=6012
ARG group_name=ubuntu
ARG group_id=6012

# create user
RUN groupadd -g 6000 data-hearai
RUN groupadd -g ${group_id} ${group_name}
RUN useradd -u ${user_id} -g ${group_id} -G 6000 -d /home/${user_name} \
    --create-home --shell /bin/bash ${user_name}
RUN echo "${user_name} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN chown -R ${user_name}:${group_name} /home/${user_name}

# user settings
ENV HOME /home/${user_name}
ENV LANG en_US.UTF-8

USER ${user_name}

CMD ["/bin/bash"]
