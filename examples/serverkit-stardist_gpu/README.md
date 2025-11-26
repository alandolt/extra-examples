![EPFL Center for Imaging logo](https://imaging.epfl.ch/resources/logo-for-gitlab.svg)
# serverkit-stardist

Implementation of a web server for [StarDist (2D)](https://github.com/stardist/stardist) with GPU support and custom model support, including caching of the model to speed up multiple requests.

To have your custom models, make a bind to the folder `/models` in the container, and place your models there. Each model should be in its own folder, containing the model weights and the coresponding `config.json` file created during training. 
The default model used if no custom model is provided is the `2D_versatile_fluo` model from the StarDist repository.

## Using `docker-compose`

Example had been set up to use `docker-compose` for easier building and running of the server, as tensorflow with GPU support is not supported anymore on all platforms by default. 

To build the docker image and run a container for the algorithm server in a single command, use:

```
docker compose up
```

The server will be running on http://localhost:8000.

## Sample images provenance

- `nuclei_2d.tif`: Fluorescence microscopy image and mask from the 2018 kaggle DSB challenge (Caicedo et al. "Nucleus segmentation across imaging experiments: the 2018 Data Science Bowl." Nature methods 16.12).