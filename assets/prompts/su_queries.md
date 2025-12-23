# Scene Understanding
```
Hey, i am doing some scene understanding and want to generate a json file that describes the scene. Can you create me an entry for each object you find? Fill in values for the placeholders in brackets "< >" The file looks like this:
[
    {
        "shape": "<shape obj1>",
        "size": "<size obj1>",
        "material": "<material obj1>",
        "color": "<color obj1>",
        "transformation": {
            "enlarges": <true|false>,
            "rotates": <true|false>,
            "orbits": <true|false>,
            "change_colors": <true|false>"
        }
    },
    {
        "shape": "<shape obj2>",
        "size": "<size obj2>",
        "material": "<material obj2>",
        "color": "<color obj2>",
        "transformation": {
            "enlarges": <true|false>,
            "rotates": <true|false>,
            "orbits": <true|false>,
            "change_colors": <true|false>"
        }
    }, ...
```

# LLM judge

### Given prediction and GT as answer
```
Hey, can you help me score the following predictions? These are two jsons, one is the ground truth json and the other is the prediction json. In the json file we have an unordered list of objects. Try to match the objects from the GT to the prediction json and write down how you would match this.
Check wherever each object is correctly identified or not from the ground truth. Increase the total score by 1 for each correctly identified object.
E.g. when color the color is cyan, the material is metal and the shape is cone in the GT and in the prediction json, the color is cyan, the material is metallic and the shape is cube, we give scores 1, 1, 0 respectively. 
Sum the scores for all objects and report the total score as a json of the form {{"score": 3}}

Ground Truth:
{}

Prediction:
{}
```