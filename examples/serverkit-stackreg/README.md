![EPFL Center for Imaging logo](https://imaging.epfl.ch/resources/logo-for-gitlab.svg)
# serverkit-pystackreg

Implementation of a web server for [pyStackreg](https://pystackreg.readthedocs.io/en/latest/index.html).

## Installing the algorithm server with `pip`

Install dependencies:

```
pip install -r requirements.txt
```

Run the server:

```
python main.py
```

The server will be running on http://localhost:8000.

## Using `docker-compose`

To build the docker image and run a container for the algorithm server in a single command, use:

```
docker compose up
```

The server will be running on http://localhost:8000.

## Sample images provenance

- pc12-unreg.tif: From the napari-pystackreg [sample image](https://github.com/glichtner/napari-pystackreg/blob/main/src/napari_pystackreg/_sample_data.py) contribution.
