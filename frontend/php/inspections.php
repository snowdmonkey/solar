<?php

include_once "rest_api.php";

$GPS_CENTER = getenv('GPS_CENTER');
$GPS_TOP = getenv('GPS_TOP');
$GPS_BOTTOM = getenv('GPS_BOTTOM');
?>
<html>
<head>
    <meta charset="UTF-8">
    <title>UAV太阳能电站巡检系统</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">


    <script type="text/javascript" src="../js/jquery.min.js"></script>

    <!--import bootstrap-->
    <link rel="stylesheet" href="../css/bootstrap.min.css">
    <script type="text/javascript" src="../js/bootstrap.min.js"></script>

    <!--import bootstap datetimepricker plugin-->
    <link rel="stylesheet" href="../css/bootstrap-datetimepicker.min.css">
    <script type="text/javascript" src="../js/moment-with-locales.js"></script>
    <script type="text/javascript" src="../js/bootstrap-datetimepicker.js"></script>

    <!--import bootstrap selectize plugin-->
    <link rel="stylesheet" href="../css/bootstrap-select.min.css">
    <script type="text/javascript" src="../js/bootstrap-select.min.js"></script>

    <!--import leaflet plugin-->
    <link rel="stylesheet" href="../css/leaflet.css">
    <script type="text/javascript" src="../js/leaflet.js"></script>

    <!--import leaflet China map provider-->
    <script type="text/javascript" src="../js/leaflet.ChineseTmsProviders.js"></script>

    <!--import leaflet vector marker plugin to customize markers-->
    <link rel="stylesheet" href="../css/leaflet-vector-markers.css">
    <script type="text/javascript" src="../js/leaflet-vector-markers.min.js"></script>

    <!--leaflet full screen control plugin-->
    <link rel="stylesheet" href="../css/Control.FullScreen.css"/>
    <script type="text/javascript" src="../js/Control.FullScreen.js"></script>

    <!--leaflet minimap control plugin-->
    <link rel="stylesheet" href="../css/Control.MiniMap.min.css"/>
    <script type="text/javascript" src="../js/Control.MiniMap.min.js"></script>

    <!--leaflet plugin for add printer-->
    <script type="text/javascript" src="../js/leaflet.browser.print.min.js"></script>

    <!--data table plugin-->
    <link rel="stylesheet" href="../css/jquery.dataTables.min.css">
    <script type="text/javascript" src="../js/jquery.dataTables.min.js"></script>


    <!--font awesome-->
    <link rel="stylesheet" href="../css/font-awesome.min.css">

    <!--<script type="text/javascript" src="../js/modernizr.custom.49511.js"></script>-->
    <!--<script type="text/javascript" src="../js/jquery.hotspot.min.js"></script>-->
    <!--<script type="text/javascript" src="../js/wheelzoom.js"></script>-->

</head>
<body>


<div class="container">
    <div class="row clearfix">
        <div class="col-md-12 column">
            <nav class="navbar navbar-default" role="navigation">
                <div class="navbar-header">
                    <button type="button" class="navbar-toggle" data-toggle="collapse"
                            data-target="#bs-example-navbar-collapse-1">
                        <span class="sr-only">Toggle navigation</span><span class="icon-bar"></span>
                        <span class="icon-bar"></span><span class="icon-bar"></span></button>
                    <a class="navbar-brand"><font color="#20b2aa">UAV太阳能电站巡检系统</font></a>
                </div>

                <div class="collapse navbar-collapse" id="bs-example-navbar-collapse-1">
                    <ul class="nav navbar-nav">
                        <li>
                            <a href="#"><font color="blue">事件查看</font></a>
                        </li>
                        <li>
                            <a href="#"><font color="blue">巡检管理</font></a>
                        </li>
                        <li class="dropdown">
                            <a href="#" class="dropdown-toggle" data-toggle="dropdown"><font
                                        color="blue">功能</font><strong class="caret"></strong></a>
                            <ul class="dropdown-menu">
                                <li>
                                    <a href="#">Action</a>
                                </li>
                                <li>
                                    <a href="#">Another action</a>
                                </li>
                                <li>
                                    <a href="#">Something else here</a>
                                </li>
                                <li class="divider">
                                </li>
                                <li>
                                    <a href="#">Separated link</a>
                                </li>
                                <li class="divider">
                                </li>
                                <li>
                                    <a href="#">One more separated link</a>
                                </li>
                            </ul>
                        </li>
                    </ul>

                    <ul class="nav navbar-nav navbar-right">
                        <li class="dropdown">
                            <form class="navbar-form navbar-left" onsubmit="return false;">
                                <div class="form-group">
                                    <input type="text" class="form-control" id="searchContent" onchange="searchDefect();"/>
                                </div>
                                <button type="button" class="btn btn-default" id="searchButton", onclick="searchDefect();">查询</button>
                            </form>
                        </li>
                        <li>
                            <a href="#"><font color="blue">帮助</font></a>
                        </li>
                    </ul>
                </div>

            </nav>
        </div>
    </div>
    <div class="row clearfix">
        <div class="col-md-3">
            <div class="pagination form-group">
                <div class="input-group date" id="datetimepicker1">
                    <input type="text" class="form-control">
                    <span class="input-group-addon">
            <span class="glyphicon glyphicon-calendar"></span>
          </span>
                </div>
            </div>
        </div>
        <script type="text/javascript">
            $(function () {
                $('#datetimepicker1').datetimepicker(
                    {
                        format: 'YYYY-MM-DD',
                        locale: "zh-CN"
                    }
                );
            });
        </script>
        <div class="col-md-9">
            <div>
                <ul class="pagination">
                    <li>
                        <a href="#">前一日</a>
                    </li>
                    <li>
                        <a href="#">6月20日</a>
                    </li>
                    <li>
                        <a href="#">6月21日</a>
                    </li>
                    <li>
                        <a>6月22日</a>
                    </li>
                    <li>
                        <a>6月23日</a>
                    </li>
                    <li>
                        <a>6月24日</a>
                    </li>
                    <li>
                        <a>7月12日</a>
                    </li>
                    <li>
                        <a>7月30日</a>
                    </li>
                    <li>
                        <a>8月10日</a>
                    </li>
                    <li>
                        <a>8月15日</a>
                    </li>
                    <li>
                        <a href="#">后一日</a>
                    </li>
                </ul>
            </div>
        </div>
    </div>
    <div class="row">
        <table id="defects" class="display">
            <thead>
            <tr>
                <th>事件编号</th>
                <th>组串编号</th>
                <th>事件状态</th>
            </tr>
            </thead>
        </table>
    </div>

    <div class="page-header"></div>

    <div class="row clearfix">
        <div class="col-md-12 column" id="layout_container">
            <div id="map" style="height: 600px;"></div>
            <div class="panel-footer"></div>
        </div>
    </div>
</div>
</body>

<script type="text/javascript">

    var mymap = L.map("map", {
        fullscreenControl: true,
        fullscreenControlOptions: {position: "topleft"}
    }).setView([<?php echo $GPS_CENTER; ?>], 18);
    L.tileLayer.chinaProvider("TianDiTu.Satellite.Map", {maxZoom: 21, minZoom: 5}).addTo(mymap);
    L.imageOverlay("../img/panorama.png",
        [[<?php echo $GPS_TOP; ?>], [<?php echo $GPS_BOTTOM; ?>]]).addTo(mymap);

    var osm2 = L.tileLayer.chinaProvider("TianDiTu.Satellite.Map", {maxZoom: 16, minZoom: 0});
    var imageOverlay2 = L.imageOverlay("../img/panorama.png",
        [[<?php echo $GPS_TOP; ?>], [<?php echo $GPS_BOTTOM; ?>]]);
    var minMapLayerGroup = L.layerGroup([osm2, imageOverlay2]);
    L.control.minimap(minMapLayerGroup, {toggleDisplay: true, minimized: false}).addTo(mymap);

    //  add leaflet printer plugin
    L.browserPrint({
        closePopupsOnPrint: true,
        printModes: ["Portrait", "Landscape", "Custom"],
        printModesNames: {"Portrait": "竖排", "Landscape": "横排", "Custom": "自定义"}
    }).addTo(mymap);

    var defectMarkers = {};
    var defectGroup = L.layerGroup().addTo(mymap);
    var cameraIcon = L.VectorMarkers.icon({icon: "camera", markerColor: "#F6F49D", prefix: "fa"});
    var unCheckedIcon = L.VectorMarkers.icon({icon: "circle", markerColor: "#466C95", prefix: "fa"});
    //      var unCheckedIcon = L.VectorMarkers.icon({markerColor: "#466C95", iconSize: [30, 30]});
    var defectIcon = L.VectorMarkers.icon({icon: "circle", markerColor: "#FF7676", prefix: "fa"});
    //      var warnIcon = L.VectorMarkers.icon({icon: "certificate", markerColor: "yellow", prefix: "glyphicon"});
    var fineIcon = L.VectorMarkers.icon({icon: "circle", markerColor: "#5DAE8B", prefix: "fa"});
    var imageGroup = L.layerGroup().addTo(mymap);
    var activeCircle = L.circle([0.0, 0.0], {radius: 5}).addTo(mymap);
    var dragStartGps = [0.0, 0.0];

    mymap.on("click", function (e) {
        imageGroup.clearLayers();
    });

    var defectTable = $("table#defects")
        .DataTable({
            "pageLength": 5,
            "dom": "rtip",
            "order": [[2, "asc"], [0, "asc"]],
            "language": {
                "sProcessing": "处理中...",
                "sLengthMenu": "显示 _MENU_ 项结果",
                "sZeroRecords": "没有匹配结果",
                "sInfo": "显示第 _START_ 至 _END_ 项结果，共 _TOTAL_ 项",
                "sInfoEmpty": "显示第 0 至 0 项结果，共 0 项",
                "sInfoFiltered": "(由 _MAX_ 项结果过滤)",
                "sInfoPostFix": "",
                "sSearch": "搜索:",
                "sUrl": "",
                "sEmptyTable": "无事件记录",
                "sLoadingRecords": "载入中...",
                "sInfoThousands": ",",
                "oPaginate": {
                    "sFirst": "首页",
                    "sPrevious": "上页",
                    "sNext": "下页",
                    "sLast": "末页"
                },
                "oAria": {
                    "sSortAscending": ": 以升序排列此列",
                    "sSortDescending": ": 以降序排列此列"
                }
            }

        });

    function updateDefectTable(defectId, category) {
        var newData;
        switch (category) {
            case 1:
                newData = "<i class='fa fa-circle' style='color:#FF7676'></i>&nbsp&nbsp异常";
                break;
            case -1:
                newData = "<i class='fa fa-circle' style='color:#5DAE8B'></i>&nbsp&nbsp正常";
        }

        defectTable.cell(function (idx, data, node) {
            return data[0] === defectId
        }, 2).data(newData);
        defectTable.draw();
    }

    function searchDefect(){
        var keyword = $("input#searchContent").val();
        defectTable.search(keyword).draw();
    }

    $("#datetimepicker1").on("dp.hide", function (e) {
        var selectedDate = e.date.format("YYYY-MM-DD");
        window.location.href = "http://"+"<?php echo getAPIHost(); ?>"+"spi/php/inspections.php?date="+btoa(selectedDate);
    });
    drawDefects();
    generateDefectTable();

    function generateDefectTable() {
        defectTable.clear();
        defectTable.draw();

        var data = '<?php getDefectsByDate($_GET['date']);?>';
        //$.get("http://"+"<?php echo getAPIHost(); ?>"+"/php/defects.php", {date: selectedDate}, function (data) {

            var defects = JSON.parse(data);

            defects.forEach(function (defect) {
                var rowData = [];
                rowData.push(defect.defectId);
                rowData.push(defect.groupId);
//        rowData.push($("<i>", {"class": "fa fa-circle", style: "color:blue"})[0].outerHTML);
                switch (defect.category) {
                    case 0:
                        rowData.push("<i class='fa fa-circle' style='color:#466C95'></i>&nbsp&nbsp疑似");
                        break;
                    case 1:
                        rowData.push("<i class='fa fa-circle' style='color:#FF7676'></i>&nbsp&nbsp异常");
                        break;
                    case -1:
                        rowData.push("<i class='fa fa-circle' style='color:#5DAE8B'></i>&nbsp&nbsp正常");
                }

                defectTable.row.add(rowData);
            });

            defectTable.draw();
        //});
    }

    defectTable.on("click", "tr", function(){
        var data = defectTable.row(this).data();
        activateMarker(defectMarkers[data[0]]);
    });

    function activateMarker(marker){
        activeCircle.setLatLng(marker.getLatLng());
    }

    function drawDefects(selectedDate) {
        defectMarkers = {};
        defectGroup.clearLayers();
        imageGroup.clearLayers();
        var data = '<?php getDefectsByDate($_GET['date']);?>';
        //$.get("http://"+"<?php echo getAPIHost(); ?>"+"/php/defects.php", {date: selectedDate}, function (data, status) {
            var defects = JSON.parse(data);

            for (var i = 0; i < defects.length; i++) {
                var defect = defects[i];

                defectMarkers[defect.defectId] =
                    L.marker([defect.latitude, defect.longitude], {draggable: true}).addTo(mymap);

                switch (defect.category) {
                    case 0:
                        defectMarkers[defect.defectId].setIcon(unCheckedIcon);
                        break;
                    case 1:
                        defectMarkers[defect.defectId].setIcon(defectIcon);
                        break;
                    case -1:
                        defectMarkers[defect.defectId].setIcon(fineIcon);
                }

                defectGroup.addLayer(defectMarkers[defect.defectId]);
            }

            defects.forEach(function (defect) {
                defectMarkers[defect.defectId].on("contextmenu", function () {
                    var that = $(this)[0];
                    var popup = L.popup({offset: [0, -30]}).setLatLng(that.getLatLng());
                    var $div = $("<div>");
                    $div.append($("<p>").append("组串编号: 000000"));
                    $div.append($("<p>").append("异常编号: " + defect["defectId"]));
                    var $confirmButton = $("<button>", {type: "button", "class": "btn btn-danger btn-sm"}).append("确认警报");
                    $confirmButton.on("click", function () {
                        that.setIcon(defectIcon);
                        popup.remove();
                        $.post("http://127.0.0.1:5000/defect/" + defect.defectId + "/category", {
                            date: selectedDate,
                            category: 1
                        }, function () {
                            updateDefectTable(defect.defectId, 1);
                        });
                    });
                    $div.append($confirmButton);
                    var $cancelButton = $("<button>", {type: "button", "class": "btn btn-success btn-sm"}).append("取消警报");
                    $cancelButton.on("click", function () {
                        that.setIcon(fineIcon);
                        popup.remove();
                        $.post("http://127.0.0.1:5000/defect/" + defect.defectId + "/category", {
                            date: selectedDate,
                            category: -1
                        }, function () {
                            updateDefectTable(defect.defectId, -1);
                        });
                    });
                    $div.append($cancelButton);
                    popup.setContent($div[0]).openOn(mymap);
                });

                defectMarkers[defect.defectId].on("click", function () {
                    imageGroup.clearLayers();
                    activateMarker(this);
                    var cameraMarkers = [];
//        $.get("http://159.99.234.54:5000/images/defect", {date: "2017-06-21", defectId: defect.defectId}, function (data) {
                    $.get("http://127.0.0.1:5000/images/defect", {
                        date: selectedDate,
                        defectId: defect.defectId
                    }, function (data) {
                        JSON.parse(data).forEach(function (item) {
                            var imageMarker = L.marker([item.latitude, item.longitude], {
                                icon: cameraIcon,
                                draggable: false
                            }).addTo(mymap);
                            imageGroup.addLayer(imageMarker);
                            cameraMarkers.push({image: item.imageName, defect: defect.defectId, marker: imageMarker});
                        });
                        cameraMarkers.forEach(function (item) {
                            var $div = $("<div>");
                            var irImgPath =
//              "http://159.99.234.54:5000/image/labeled?image="
                                "http://127.0.0.1:5000/image/labeled?image="
                                + item.image + "\&defect=" + item.defect + "\&date=" + selectedDate;
                            var $irImg = $("<img>", {src: irImgPath});
//            var visImgPath = "http://159.99.234.54:5000/image/visual?image=" + item.image + "\&date="+"2017-06-21";
                            var visImgPath = "http://127.0.0.1:5000/image/visual?image=" + item.image + "\&date=" + selectedDate;
                            var $visImg = $("<img>", {src: visImgPath});
                            var $ul = $("<ul>", {"class": "nav nav-pills"}).append(
                                $("<li>", {"class": "active"}).append(
                                    $("<a>", {"data-toggle": "tab", "href": "#irImageTab"}).append("红外照片")
                                )
                            ).append(
                                $("<li>").append(
                                    $("<a>", {"data-toggle": "tab", "href": "#visImageTab"}).append("可见光照片")
                                )
                            );
                            $div.append($ul).append(
                                $("<div>", {"class": "tab-content"}).append(
                                    $("<div>", {id: "irImageTab", "class": "row tab-pane fab in active"}).append($irImg)
                                ).append(
                                    $("<div>", {id: "visImageTab", "class": "row tab-pane fad"}).append($visImg)
                                )
                            );

                            item.marker.on("click", function () {
                                var popup = L.popup({offset: [0, -30]}).setContent($div[0]);
                                popup.setLatLng(item.marker.getLatLng());
                                mymap.addLayer(popup);
                                var pos = mymap.latLngToLayerPoint(popup._latlng);
                                L.DomUtil.setPosition(popup._wrapper.parentNode, pos);
                                var draggable = new L.Draggable(popup._container, popup._wrapper);
                                draggable.enable();
                            });
                        });
                    });
                });

                defectMarkers[defect.defectId].on("dragstart", function () {
                    dragStartGps = this.getLatLng();
                });

                defectMarkers[defect.defectId].on("dragend", function () {
                    var r = confirm("确认移动异常点至该位置？\n如果确认，该移动信息将用于改善定位服务。");
                    if (r === false) {
                        this.setLatLng(dragStartGps);
                    } else {
                        $.post("http://127.0.0.1:5000/defect/" + defect.defectId + "/position",
                            {lat: this.getLatLng().lat, lng: this.getLatLng().lng, date: selectedDate});
                    }
                })
            })
    }


</script>

</html>