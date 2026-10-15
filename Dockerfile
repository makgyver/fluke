# start from this pythorch image as a base as it supports data serialization
FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime 
RUN pip install fluke-fl

# Install git
RUN apt-get update && apt-get install -y \
    software-properties-common
RUN add-apt-repository universe
RUN apt-get install python3-git -y

WORKDIR /fluke

# Mount volume for datasets
ENV EXP_CONFIG=""
ENV ALG_CONFig=""
ENV MODE=""

ENTRYPOINT [ "sh", "-c", "wandb login ${API}; fluke-get config ${EXP_CONFIG}; fluke-get config ${ALG_CONFig}; fluke --config=config/${EXP_CONFIG}.yaml ${MODE} config/${ALG_CONFig}.yaml" ]