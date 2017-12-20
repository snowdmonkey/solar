conn = new Mongo();
db = conn.getDB("solar");

db.createCollection("exif", {
  capped: true,
  autoIndexId: false,
  size: 1000000
});

db.exif.createIndex({station: 1, date: -1});

db.createCollection("defect", {
  capped: true,
  autoIndexId: false,
  size: 1000000
});

db.defect.createIndex({station: 1, date: -1});

db.createCollection("log", {
  capped: true,
  autoIndexId: false,
  size: 1000000
});

db.log.createIndex({jobId: 1, timestamp: -1});

db.createCollection("panelGroup", {
  capped: true,
  autoIndexId: false,
  size: 1000000
});

db.panelGroup.createIndex({station: 1});

db.createCollection("rect", {
  capped: true,
  autoIndexId: false,
  size: 1000000
});


db.rect.createIndex({station: 1, date: -1});

db.createCollection("station", {
  capped: true,
  autoIndexId: false,
  size: 1000000
});

db.station.createIndex({stationId: 1});