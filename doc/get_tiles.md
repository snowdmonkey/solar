# Get tiles from panorama

The Python module to generate tiles is `/backend/prototype/mapping/tiles.py`

To use it, starts from repository root directory
* On Windows

```bash
set PYTHONPATH=%cd%\backend\prototype
```
* On Linux

```bash
export PYTHONPATH=$(pwd)/backend/prototype
```
You can type

```bash
python -m mapping.tiles -h
```
to get the usage of the module.
##Example usage on Linux
Suppose `PANORAMA_PATH` is the path for the stitch GeoTiff file.

```bash
mkdir output
cd output
python -m mapping.tiles $(PANORAMA_PATH)
```
Then tile files will be created under `output/tiles`. `output/position.json` is also created to record the GPS 
information for each tile file.

When deploy a new station, both output/tiles/*.png and output/position.json need to be copied to 
`/$(ROOT)/spi/$(CUSTOMER_ID)/tiles/$(STATION_ID)`.

It usage on Windows machine is similar.