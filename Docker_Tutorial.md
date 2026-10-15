Image to run experiments in a containerized way on the fluke federated learning framework.

## V1.0: allows to use the latest version of the fluke package to run your experiments
```bash
docker run --env EXP_CONFIG="<experiment>" --env ALG_CONFIG="<algortihm>" --env MODE="<mode>" --env API="<api key>" -v $path_to_configs:/fluke/config -v $(path_to_data):/fluke/data 00uno00/fluke_benchmark:1.0
```

## V1.1: allows to use local version of the fluke framework to run your experiments
```bash
docker run --env FILE="<python script>" --env API="<api key>" -v $path_to_fluke:/fluke 00uno00/fluke_benchmark:1.1
```

where:
- EXP_CONFIG is your experiment configuration (base one is exp.yaml)
- ALG_CONFIG is the algorithm configuration 
- MODE is the mode in which to run the experiment (either federation, centralized or client-side)
- FILE is the python script in which the [experiment](https://makgyver.github.io/fluke/tutorials.html) is set up **WARNING: this file must be in the fluke general repository** (only for V1.1)
- API is your wandb api key (not necessary)

If either configurations is not present in the default config don't worry, the "get" commands will fail but the configuration will still run.
## Testing the image
you can run this command as a test for V1.0:
```bash
docker run --env EXP_CONFIG="exp" --env ALG_CONFIG="fedavg" --env MODE="federation" --env API="" -v path_to_configs:/fluke/config -v path_to_data:/fluke/data 00uno00/fluke_benchmark:1.0
```

or for V1.1:
```bash
docker run --env FILE="test.py" --env API="" -v path_to_fluke:/fluke 00uno00/fluke_benchmark:1.1
```

*taken from https://hub.docker.com/r/00uno00/fluke_benchmark*