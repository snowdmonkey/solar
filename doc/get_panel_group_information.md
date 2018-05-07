#Extract panel group GPS information

The Python module to generate tiles is `/backend/prototype/mapping/label_panel_group.py`

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
python -m mapping.label_panel_group.py -h
```
to get the usage of the module.

##Example usage on Linux
1. Suppose `PANORAMA_PATH` is the path to the stitch GeoTiff file. 
```bash
mkdir output
cd output
python -m mapping.label_panel_group tosvg $(PANORAMA_PATH)
```
Then `output/panelgroup_auto.svg` is created. 

2. Open GeoTiff file in `GIMP` (https://www.gimp.org/).

![gimp](doc/img/gimp.png)

3. import output/panelgroup_auto.svg as path.

![gimp_auto](doc/img/gimp_auto.png)

4. Manually correct the paths.

![gimp_maul](doc/img/gimp_manul.png)

5. Merge all the paths and export it as `output/panelgroup_manual.svg`.

6. from folder `output`
```bash
python -m mapping.label_panel_group fromsvg gimp_manual.svg 
```

File `output/groupPanel.json` is created, copy it to `$(ROOT)/spi/$(CUSTOMER_ID)/inspection/$(STATION_ID)/groupPanel.json`.



It usage on Windows machine is similar.