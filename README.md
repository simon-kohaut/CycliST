# CycliST: A Benchmark for Visual Reasoning on Cyclical State Transitions

## Installation

First, some dependencies need to be installed.
You can do so by running the following commands.

```python
# Placeholder
```

Once the dependencies are installed, you can setup your local installation of CycliST with Python.
To do so, simply run `pip install .` or (if you have not already cloned this repository) `pip install git+https://github.com/simon-kohaut/CycliST.git`.

## Dataset Generation

CycliST is created in a two-step process.
First, randomized scenes are initialized, containing objects and cycles within predefined limits.
Then, we render the scenes out as videos using Blender 4.0.
Subsequently, the generated scene is labelled to create question-answer-pairs alongside the video.

To run this process manually, you can use CycliST as a command line tool.
For example, a simple scene may be rendered by running:

```bash
cyclist --device CUDA --num_samples 1 --min_objects 2 --max_objects 5 --min_duration 5 --max_duration 10 --fps 32 --split ExampleSplit
```

This will render out a randomized scene given the provided settings in CycliST's output/videos folder.
Similarly, scene descritptions are placed in the output/scenes folder and question-answer-pairs in the output/questions folder.

## Acknowledgments

Parts of this repository is in parts based on and extends ideas from [CLEVR](https://github.com/facebookresearch/clevr-dataset-gen).


## Docker installation
You can create a container with the following docker image:
```hansiwusti/llava-changes:1.0```
Inside the container run the install.sh to download blender and its libraries.



## SG Lang installation
```. install_packages.sh ``` -> results in 
``` pip install -e .  ```
``` . install_sg_lang.sh ``` -> results in missing library
