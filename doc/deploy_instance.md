# SPI Portal安装指南

1. 安装Docker

SPI Portal应用，一下简称应用，可部署在任何支持Docker容器运行环境的服务器或云服务中。本指南以CentOS 7.3操作系统环境为例，说明应用的安装、配置过程。
在服务器上安装Docker相关内容可参考相关链接：https://www.docker.com/get-docker

2. 创建目录结构

在服务器文件系统中构建如下目录结构：

![gimp](doc/img/hierachy.png)

3. 更新部署脚本
在工作目录`$WORKDIR`中进行如下操作：

    1. 获取后端代码：

```bash
git clone http://hcelab.honeywell.com.cn/gitlab/uav/uav-solar-panel.git
```

    2. 获取前端代码：

```bash
git clone http://hcelab.honeywell.com.cn/gitlab/solar-panel-ui/web-app.git
```
    3. 根据实际替换{}和相关内容，并保存在工作目录中，例如将{customerid}替换为linuo
   
将替换完成的{customerid}.conf文件移动到$NGINXDIR，即$ROOT/spi/conf/nginx目录中。
4. 构建镜像
    1. spiproxy

```bash
cd $WORKDIR/web-app/nginx
docker build -t spiproxy .
```

    2.
```bash
solarapi & solarui:{customerid}
cd $WORKDIR
./app_ctl.sh buildall
```
5.	启动应用
    1. 启动spiproxy

```bash
docker run --name spiproxy --restart=always -d -p 80:80 -v $NGINXDIR:/etc/nginx/conf.d/brands spiproxy
```
    2. 启动应用

```bash
cd $WORKDIR
./app_ctl.sh start
```