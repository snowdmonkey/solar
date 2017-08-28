from flask import Flask
import json

app = Flask(__name__)

defects_summary = None


def get_defects_summary() -> dict:
    global defects_summary
    if defects_summary is None:
        with open("C:/Users/h232559/Documents/projects/uav/pic/2017-06-21-funingyilin-DJI/6-21-FLIR/defects.json") as f:
            defects_summary = json.load(f)
    return defects_summary


@app.route("/defects")
def get_defects():
    defects = list()
    for defect_id, defect_info in get_defects_summary().items():
        defect = {"defectId": defect_id,
                  "latitude": defect_info.get("latitude"),
                  "longitude": defect_info.get("longitude")}
        defects.append(defect)
    return json.dumps(defects)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)