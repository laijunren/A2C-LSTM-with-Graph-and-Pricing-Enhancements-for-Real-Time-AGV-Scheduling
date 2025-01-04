@echo off
rem 
rem Run AnyLogic Experiment
rem  
set DIR_BACKUP_XJAL=%cd%
cd /D "%~dp0"

rem ----------------------------------

set SIM_API_HOST=http://localhost:5011
set SIM_ACCESS_KEY=00000
set SIM_WORKSPACE=rl-hospital

set JAVA_BIN="D:\my_code\Java\jdk-11\bin\java.exe"


FOR /F "usebackq tokens=4" %%A IN (`java -fullversion 2^>^&1`) DO (
	set VERSION=%%~A
)

echo Java version: %VERSION%

set OPTIONS_XJAL=--illegal-access=deny
IF "%VERSION:~0,2%"=="1." set OPTIONS_XJAL=

rem ---------------------------

echo on

%JAVA_BIN% %OPTIONS_XJAL% -cp model.jar;lib/MarkupDescriptors.jar;lib/PedestrianLibrary.jar;lib/ProcessModelingLibrary.jar;lib/Core.jar;lib/LevelSystem.jar;lib/NavigationMesh.jar;lib/MaterialHandlingLibrary.jar;lib/RackSystemModule.jar;lib/heuristic.jar;lib/jts-core-1.16.0.jar;lib/com.anylogic.engine.jar;lib/com.anylogic.engine.nl.jar;lib/com.anylogic.engine.sa.jar;lib/sa/com.anylogic.engine.sa.web.jar;lib/sa/executor-basic-8.3.jar;lib/sa/ioutil-8.3.jar;lib/sa/jackson/jackson-annotations-2.12.2.jar;lib/sa/jackson/jackson-core-2.12.2.jar;lib/sa/jackson/jackson-databind-2.12.2.jar;lib/sa/spark/javax.servlet-api-3.1.0.jar;lib/sa/spark/jetty-client-9.4.31.v20200723.jar;lib/sa/spark/jetty-continuation-9.4.31.v20200723.jar;lib/sa/spark/jetty-http-9.4.31.v20200723.jar;lib/sa/spark/jetty-io-9.4.31.v20200723.jar;lib/sa/spark/jetty-security-9.4.31.v20200723.jar;lib/sa/spark/jetty-server-9.4.31.v20200723.jar;lib/sa/spark/jetty-servlet-9.4.31.v20200723.jar;lib/sa/spark/jetty-servlets-9.4.31.v20200723.jar;lib/sa/spark/jetty-util-9.4.31.v20200723.jar;lib/sa/spark/jetty-webapp-9.4.31.v20200723.jar;lib/sa/spark/jetty-xml-9.4.31.v20200723.jar;lib/sa/spark/slf4j-api-1.7.25.jar;lib/sa/spark/spark-core-2.9.3.jar;lib/sa/spark/websocket-api-9.4.31.v20200723.jar;lib/sa/spark/websocket-client-9.4.31.v20200723.jar;lib/sa/spark/websocket-common-9.4.31.v20200723.jar;lib/sa/spark/websocket-server-9.4.31.v20200723.jar;lib/sa/spark/websocket-servlet-9.4.31.v20200723.jar;lib/sa/util-8.3.jar;lib/database/querydsl/querydsl-core-4.2.1.jar;lib/database/querydsl/querydsl-sql-4.2.1.jar;lib/database/querydsl/querydsl-sql-codegen-4.2.1.jar;lib/poi/dom4j-1.6.1.jar;lib/poi/poi-3.10.1-20140818.jar;lib/poi/poi-examples-3.10.1-20140818.jar;lib/poi/poi-excelant-3.10.1-20140818.jar;lib/poi/poi-ooxml-3.10.1-20140818.jar;lib/poi/poi-ooxml-schemas-3.10.1-20140818.jar;lib/poi/poi-scratchpad-3.10.1-20140818.jar;lib/poi/stax-api-1.0.1.jar;lib/poi/xmlbeans-2.6.0.jar;lib/database/querydsl/guava-18.0.jar;lib/com.anylogic.rl.data.jar;lib/com.anylogic.rl.connector.support.jar;lib/com.anylogic.rl.connector.bonsai.jar -Xmx512m hospital_DRL_training.RLExperiment %*

@echo off
if %ERRORLEVEL% neq 0 pause
:end
echo on
@cd /D "%DIR_BACKUP_XJAL%"
