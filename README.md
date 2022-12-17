# fl-bench
Python module to benchmark federated learning algorithms

## TODO ASAP
- [ ] Check the correctness of SCAFFOLD and FedBN
- [ ] FedSGD: add support to `batch_size != 0`, i.e., the client can perform a local update on a subset (the only batch!) of the data
- [ ] Test logging on wandb
- [ ] Add support to validation
- [ ] Add client-side evaluations
- [ ] Add documentation

## DESIDERATA
- [ ] Add more algorithms
- [ ] Add more datasets
- [ ] Add support to tensorboard