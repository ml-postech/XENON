<div align="center">
<h2 align="center">
    <b> XENON: Experience-based Knowledge Correction
     <br /> for Robust Planning in Minecraft
   <br /> <font size=4>ICLR 2026 </font></b> 
</h2>
<div>
<a target="_blank" href="https://scholar.google.com/citations?user=9665NJ8AAAAJ">Seungjoon&#160;Lee</a><sup>1</sup>,
<a target="_blank" href="https://scholar.google.com/citations?user=iNjrWK0AAAAJ">Suhwan&#160;Kim</a><sup>1</sup>,
<a target="_blank" href="https://scholar.google.com/citations?user=btj5roEAAAAJ">Minhyeon&#160;Oh</a><sup>1</sup>,
<a target="_blank" href="https://scholar.google.com/citations?user=s82gfdAAAAAJ">Youngsik&#160;Yoon</a><sup>1</sup>,
 <a target="_blank" href="https://scholar.google.com/citations?user=KWG3UUMAAAAJ">Jungseul&#160;Ok</a><sup>1 2 &#9993</sup>
</div>
<sup>1</sup>Department of Computer Science & Engineering, POSTECH
<br />
<sup>2</sup>Graduate School of Artificial Intelligence, POSTECH
<br />
<sup>&#9993&#160;</sup>Corresponding author&#160;&#160;
<br/>
<div align="center">
    <a href="https://openreview.net/forum?id=N22lDHYrXe" target="_blank">
    <img src="https://img.shields.io/badge/Paper-OpenReview-deepgreen" alt="Paper OpenReview"></a>
    <a href="https://sjlee-me.github.io/XENON/" target="_blank">
    <img src="https://img.shields.io/badge/Project_Page-XENON-blue" alt="Project Page"></a>
</div>
</div>



## :new: Updates
<!-- - [11/2024] :fire: We have created a repository to track the latest advancements in [Minecraft Agents](https://github.com/dawn0815/Awesome-Minecraft-Agent).
- [10/2024] :fire: We release the presentation [video](https://youtu.be/SWnGs3TXRp0) and [demo](https://youtu.be/NgfDbEdACS8). -->
- [01/2026] :fire: We release the code.
- [01/2026] :fire: XENON is accepted to [**ICLR 2026**](https://openreview.net/forum?id=N22lDHYrXe)!
<!-- - [08/2024] :fire: [Project page](https://cybertronagent.github.io/Optimus-1.github.io/) released.
- [08/2024] :fire: [Arxiv paper](https://arxiv.org/abs/2408.03615) released. -->

## How to run XENON

### Environment setup

0. Install STEVE-1 checkpoint (Thanks [Optimus-1](https://github.com/JiuTian-VL/Optimus-1) !)

```shell
# url: https://drive.google.com/file/d/1Mmwqv2juxMuP1xOZYWucnbKopMk0c0DV/view?usp=drive_link
pip install gdown
gdown 1Mmwqv2juxMuP1xOZYWucnbKopMk0c0DV

# After the download is done
unzip optimus1_steve1_ckpt.zip
# Then, `checkpoints` directory should be made at `{path/to/this/repo}/checkpoints`
```

1. Install MCP-Reborn (Again, thanks [Optimus-1](https://github.com/JiuTian-VL/Optimus-1) !)
```shell
# url: https://drive.google.com/file/d/1GLy9IpFq5CQOubH7q60UhYCvD6nwU_YG/view?usp=drive_link
gdown 1GLy9IpFq5CQOubH7q60UhYCvD6nwU_YG

# After the download is done
mv MCP-Reborn.tar.gz minerl/minerl
```

2. Install docker image (or build your docker image using the Dockerfile in our `dev_docker` directory):
```shell
docker pull sjlee1218/xenon:latest
```

3. Run a docker container
```shell
# mount this respository into a container's /app/repo, and mount your HF_HOME into a container's /app/LLM
docker run --rm --gpus '"device=0,1"' \
 -v {path/to/this/repo}:/app/repo \
 -v ~/.cache/huggingface:/app/LLM \
 --shm-size=32g \
 -it sjlee1218/xenon:latest  /bin/bash
```

4. Compile MCP-Reborn inside the docker container
```shell
# Inside the docker container
cd /app/repo/minerl/minerl
rm -rf MCP-Reborn
tar -xzvf MCP-Reborn.tar.gz --no-same-owner
cd MCP-Reborn
./gradlew clean build shadowJar
```

Then you can run XENON!

### Running experiments

0. Run a docker container

```shell
# mount this respository into a container's /app/repo, and mount your HF_HOME into a container's /app/LLM
docker run --rm --gpus '"device=0,1"' \
 -v {path/to/this/repo}:/app/repo \
 -v ~/.cache/huggingface:/app/LLM \
 --shm-size=32g \
 -it sjlee1218/xenon:latest  /bin/bash
```

1. [Planning] Run XENON to solve long-horizon goals, given a oracle dependency graph (i.e. correct recipes are given)
```shell
cd /app/repo
export HF_HOME="/app/LLM"

xvfb-run -a python app.py --port 9000 > /dev/null 2>&1 &
sleep 3

xvfb-run -a python -m optimus1.main_planning server.port=9000 env.times=1 benchmark=wooden evaluate="[0]" env.times=1 prefix="ours_planning"

python -m optimus1.util.server_api --port 9000 # shutdown the app.py server, after an experiment is done
```
Or do `bash scripts/run_planning_diamond.sh` if you want to run many experiments by one command.


2. [Learning] Run XENON to learn dependencies and actions for goals
```shell
cd /app/repo
export HF_HOME="/app/LLM"

xvfb-run -a python app.py --port 9000 > /dev/null 2>&1 &
sleep 3

python -m optimus1.main_exploration server.port=9000 env.times=1 prefix="ours_exploration"

python -m optimus1.util.server_api --port 9000 # shutdown the app.py server
```
Or do `bash scripts/run_exploration.sh` if you want to run many experiments by one command.



<!-- ## :balloon: Optimus-1 Framework
We divide the structure of Optimus-1 into Knowledge-Guided Planner, Experience-Driven Reflector, and Action Controller. In a given game environment with a long-horizon task, the Knowledge-Guided Planner senses the environment, retrieves knowledge from HDKG, and decomposes the task into executable sub-goals. The action controller then sequentially executes these sub-goals. During execution, the Experience-Driven Reflector is activated periodically, leveraging historical experience from AMEP to assess whether Optimus-1 can complete the current sub-goal. If not, it instructs the Knowledge-Guided Planner to revise its plan. Through iterative interaction with the environment,Optimus-1 ultimately completes the task.
<img src="./assets/fig2.png" >

## :smile_cat: Evaluation results
We report the `average success rate (SR)`, `average number of steps (AS)`, and `average time (AT)` on each task group, the results of each task can be found in the Appendix experiment. Lower AS and AT metrics mean that the agent is more efficient at completing the task, while $∞$ indicates that the agent is unable to complete the task. Overall represents the average result on the five groups of Iron, Gold, Diamond, Redstone, and Armor.
<img src="./assets/table1.png" >
 -->

## :hugs: Citation

If you find this work useful for your research, please kindly cite our paper:

```
@inproceedings{lee2026experience,
    title={Experience-based Knowledge Correction for Robust Planning in Minecraft},
    author={Seungjoon Lee and Suhwan Kim and Minhyeon Oh and Youngsik Yoon and Jungseul Ok},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=N22lDHYrXe}
}
```

## Acknowledgement
- Our codebase is largely inspired by [Optimus-1](https://arxiv.org/abs/2408.03615v2)'s official [github repository](https://github.com/JiuTian-VL/Optimus-1).
- Thanks for these awesome minecraft agents: [DECKARD](https://arxiv.org/abs/2301.12050), [STEVE-1](https://arxiv.org/abs/2306.00937), [ADAM](https://arxiv.org/abs/2410.22194), etc.
