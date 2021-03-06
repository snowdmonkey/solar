<html>
<head>
    <meta charset="UTF-8">
    <title>UAV太阳能电站巡检系统</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="../css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="../css/jquery.hotspot.css">
    <link rel="stylesheet" href="../css/leaflet.css">
    <link rel="stylesheet" href="../css/bootstrap.min.css">
    <link rel="stylesheet" href="../css/bootstrap-datetimepicker.min.css">
    <link rel="stylesheet" href="../css/bootstrap-select.min.css">
    <script type="text/javascript" src="../js/jquery.min.js"></script>
    <script type="text/javascript" src="../js/bootstrap.min.js"></script>
    <script type="text/javascript" src="../js/modernizr.custom.49511.js"></script>
    <script type="text/javascript" src="../js/jquery.hotspot.min.js"></script>
    <script type="text/javascript" src="../js/bootstrap-select.min.js"></script>
    <script type="text/javascript" src="../js/moment.js"></script>
    <script type="text/javascript" src="../js/bootstrap-datetimepicker.js"></script>
    <script type="text/javascript" src="../js/leaflet.js"></script>
    <script type="text/javascript" src="../js/leaflet.ChineseTmsProviders.js"></script>
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
                            <a href="../html/index.html"><font color="blue">事件查看</font></a>
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
                            <form class="navbar-form navbar-left" role="search">
                                <div class="form-group">
                                    <input type="text" class="form-control"/>
                                </div>
                                <button type="submit" class="btn btn-default">查询</button>
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
                        format: 'YYYY/MM/DD'
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
    <div class="row clearfix">
        <div class="col-md-6 column">
            <div class="list-group">
                <a class="list-group-item active">巡查事件<span class="badge">3</span></a>

                <div class="list-group-item">
                    <font size="4" color="black">&bull;&nbsp;&nbsp;</font><a href="#">面板标号#0020122</a>
                </div>
                <div class="list-group-item">
                    <font size="4" color="black">&bull;&nbsp;&nbsp;</font><a href="#">面板标号#0070692</a>
                </div>
                <div class="list-group-item">
                    <font size="4" color="black">&bull;&nbsp;&nbsp;</font><a href="#">面板标号#0030661</a>
                </div>
                <a class="list-group-item active"> </a>
            </div>
        </div>
        <div class="col-md-6 column">
            <div class="list-group">
                <a class="list-group-item active">未处理工单<span class="badge">3</span></a>

                <div class="list-group-item">
                    <font size="4" color="black">&bull;&nbsp;&nbsp;</font><a href="#">工单编号#0069011</a>
                </div>
                <div class="list-group-item">
                    <font size="4" color="black">&bull;&nbsp;&nbsp;</font><a href="#">工单编号#0069016</a>
                </div>
                <div class="list-group-item">
                    <font size="4" color="black">&bull;&nbsp;&nbsp;</font><a href="#">工单编号#0069018</a>
                </div>
                <a class="list-group-item active"> </a>
            </div>
        </div>
    </div>
    <div class="row clearfix">
        <div class="col-md-12 column" id="layout_container">
            <div class="popover right pop1" data-easein="cardInRight" data-easeout="cardOutRight" id="pop1">
                <div class="arrow"></div>
                <div class="popover-inner">
                    <h3 class="popover-title">面板详情</h3>
                    <div class="popover-content">
                        <p>面板标号#0020122</p>
                        <img src="../img/DJI_0104.jpg" alt="small" width="600"/>
                    </div>
                </div>
            </div>
            <div class="popover top pop2" data-easein="cardInTop" data-easeout="cardOutTop" id="pop2">
                <div class="arrow"></div>
                <div class="popover-inner">
                    <h3 class="popover-title">面板详情</h3>
                    <div class="popover-content">
                        <p>面板标号#0070692</p>
                        <img src="../img/DJI_0107.jpg" alt="small" width="600"/>
                    </div>
                </div>
            </div>
            <div class="popover left pop3" data-easein="cardInLeft" data-easeout="cardOutLeft" id="pop3">
                <div class="arrow"></div>
                <div class="popover-inner">
                    <h3 class="popover-title">面板详情</h3>
                    <div class="popover-content">
                        <p>面板标号#0030661</p>
                        <img src="../img/DJI_0119.jpg" alt="small" width="600"/>
                    </div>
                </div>
            </div>
            <!--<div class="jumbotron">-->
            <!--<img src="../img/arrow1.png" alt="info" class="info-icon info-icon1" data-target="pop1"/>-->
            <!--<img src="../img/arrow1.png" alt="info" class="info-icon info-icon2" data-target="pop2"/>-->
            <!--<img src="../img/arrow1.png" alt="info" class="info-icon info-icon3" data-target="pop3"/>-->

            <!--<img class="jumbotron" src="../img/panorama.png" width="100%" id="image2" usemap="#farmmap">-->

            <!--<map name="farmmap">-->
            <!--<area shape="circle" coords="300,300,10" onclick="alert('oops')" href="#"/>-->
            <!--</map>-->

            <div id="map" style="height: 600px;"></div>
            <div class="panel-footer"></div>
        </div>
    </div>
</div>
</body>

<script type="text/javascript">

    var mymap = L.map("map").setView([33.58873250, 119.6334535], 18);
    L.tileLayer.chinaProvider("TianDiTu.Satellite.Map", {maxZoom: 20, minZoom: 5}).addTo(mymap);
    L.imageOverlay("../img/panorama.png",
        [[33.585694321877583, 119.63701667124482], [33.591771424236896, 119.6298894339357]]).addTo(mymap);

    var defectMarkers = {};
    var imageGroup = L.layerGroup().addTo(mymap);
    $.get("http://127.0.0.1:5000/defects", function(data, status){
        var defects = JSON.parse(data);

        for(var i = 0; i < defects.length; i++){
            var defect = defects[i];
            defectMarkers[defect.defectId] = L.marker([defect.latitude, defect.longitude]).addTo(mymap);
        }

        defects.forEach(function(defect){
            $.get("http://127.0.0.1:5000/images/defect/"+defect.defectId, function(data){
                defectMarkers[defect.defectId].on("click", function(){
                    imageGroup.clearLayers();
                    JSON.parse(data).forEach(function(item){
                        var imageMarker = L.marker([item.latitude, item.longitude]).addTo(mymap);
                        imageGroup.addLayer(imageMarker);
                    })
                })
            })
        })

    })

</script>

</html>