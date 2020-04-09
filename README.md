<div align="center">

[![Catalyst logo](https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/pics/catalyst_logo.png)](https://github.com/catalyst-team/catalyst)

**Accelerated DL & RL**

[![Build Status](http://66.248.205.49:8111/app/rest/builds/buildType:id:Catalyst_Deploy/statusIcon.svg)](http://66.248.205.49:8111/project.html?projectId=Catalyst&tab=projectOverview&guest=1)
[![CodeFactor](https://www.codefactor.io/repository/github/catalyst-team/catalyst/badge)](https://www.codefactor.io/repository/github/catalyst-team/catalyst)
[![Pipi version](https://img.shields.io/pypi/v/catalyst.svg)](https://pypi.org/project/catalyst/)
[![Docs](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fcatalyst%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)](https://catalyst-team.github.io/catalyst/index.html)
[![PyPI Status](https://pepy.tech/badge/catalyst)](https://pepy.tech/project/catalyst)

[![Twitter](https://img.shields.io/badge/news-on%20twitter-499feb)](https://twitter.com/catalyst_core)
[![Telegram](https://img.shields.io/badge/channel-on%20telegram-blue)](https://t.me/catalyst_team)
[![Slack](https://img.shields.io/badge/Catalyst-slack-success)](https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw)
[![Github contributors](https://img.shields.io/github/contributors/catalyst-team/catalyst.svg?logo=github&logoColor=white)](https://github.com/catalyst-team/catalyst/graphs/contributors)


</div>

PyTorch framework for DL & RL research.
It was developed with a focus on reproducibility,
fast experimentation and code/ideas reusing.
Being able to research/develop something new,
rather than write another regular train loop. <br/>
Break the cycle - use the Catalyst!

Part of [PyTorch Ecosystem](https://pytorch.org/ecosystem/). Part of [Catalyst Ecosystem](https://docs.google.com/presentation/d/1D-yhVOg6OXzjo9K_-IS5vSHLPIUxp1PEkFGnpRcNCNU/edit?usp=sharing). Project [manifest](https://github.com/catalyst-team/catalyst/blob/master/MANIFEST.md).

---

# Catalyst.Gan [WIP]  [![Github contributors](https://img.shields.io/github/contributors/catalyst-team/segmentation.svg?logo=github&logoColor=white)](https://github.com/catalyst-team/gan/graphs/contributors)

You will learn how to train your GAN using the Catalyst framework.
The main advantage is to customize your experiments in the yaml config instead of the code.

# Installation

```bash
pip install -r requirements.txt
```

# Run examples

```bash
# (Goodfellow et. al., 2014: https://arxiv.org/pdf/1406.2661.pdf)
catalyst-dl run --config=./gan_example/configs/vanilla_gan.yml --verbose
# (Arjovsky et. al., 2017: https://arxiv.org/abs/1701.07875)
catalyst-dl run --config=./gan_example/configs/wasserstein_gan.yml --verbose
# (Gulrahani et. al., 2017: https://arxiv.org/abs/1704.00028)
catalyst-dl run --config=./gan_example/configs/wasserstein_gan_gp.yml --verbose
# (Mirza and Osindero, 2014: https://arxiv.org/abs/1411.1784)
catalyst-dl run --config=./gan_example/configs/conditional_gan.yml --verbose
```
