# Contributing to `fluke`

[fork]: https://github.com/makgyver/fluke/fork
[pr]: https://github.com/makgyver/fluke/compare
[style]: https://peps.python.org/pep-0008/
[code-of-conduct]: https://github.com/makgyver/fluke/blob/main/CODE_OF_CONDUCT.md

First of all, thank you for considering contributing to `fluke`! We are happy to have you here.

Please note that this project is released with a [Contributor Code of Conduct][code-of-conduct].
By participating in this project you agree to abide by its terms.

## Issues and Pull Requests

If you have suggestions for how this project could be improved, or want to report a bug, open an issue! We'd love all and any contributions. If you have questions, too, we'd love to hear them.

We'd also love PRs. If you're thinking of a large PR, we advise opening up an issue first to talk about it, though! Look at the links below if you're not sure how to open a PR.

### Submitting a pull request

1. [Fork][fork] and clone the repository.
1. Create a new virtual env: `python -m venv venv` (or whatever you wanna call it) and activate it: `source venv/bin/activate`.
1. Install the dependencies: `pip install -r requirements.txt`.
1. If everything is installed correctly, you should be able to run fluke by typing `python -m fluke.run` in the terminal.
1. Make sure the tests pass on your machine: `pytest`, note: you may need to install `pytest` first.
1. Create a new branch: `git checkout -b my-branch-name`.
1. Make your change, add tests, and make sure the tests still pass.
1. Push to your fork and [submit a pull request][pr].
1. Pat yourself on the back :) and wait for your pull request to be reviewed and merged.

Here are a few things you can do that will increase the likelihood of your pull request being accepted:

- Follow the PEP8 [style guide][style].
- Write and update tests.
- Keep your changes as focused as possible. If there are multiple changes you would like to make that are not dependent upon each other, consider submitting them as separate pull requests.
- Write a [good commit message](http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html).


## Requesting a new feature

If you have an idea for a new feature, please [open an issue](https://github.com/makgyver/fluke/issues/new?assignees=&labels=&projects=&template=feature_request.md&title=") and describe the feature you would like to see implemented following the provided template. We will discuss the feature and decide whether it is a good fit for the project. If it is, we will add it to the project's roadmap and you can start working on it.


