#! /bin/bash
#Sample bash script to automate data download via PRADAN. 
#Windows users may install wget.exe and write a batch script in the same lines.
#Prequisites: Login to Pradan in your browser, select data of your interest and download script for the session
#Caution: There are session download limits, request rate limit and session timeouts in place, etc.
#	Violations may lead to blocking. Use script to ease the manual data download efforts but do not load the server.

cookies="FGTServer=03DE191863F4388C06A7AAAF7E0136FBD15060DF21FA637D82A675307CD5BF28BF8658CAFD950178C9994D;primefaces.download=null;FGTServer=03DE191863F4388C06A7AAAF7E0136FBD15060DF21FA637D82A675307CD5BF28BF8658CAFD950178C9994D;JSESSIONID=7348037193e7dd0bae01e75dd1e1;JSESSIONID=aa780b42b3b5c148c87dadae67a2;OAuth_Token_Request_State=706caef0-8a32-4a0f-ba61-a0c099cb2f09;"
urlPrefix="https://pradan1.issdc.gov.in"
#proxyOptions are required if your organization uses proxy to connect to Internet.
#proxyOptions="-e use_proxy=yes -e https_proxy=127.0.0.1:8080"
proxyOptions=""

#keepalive for 1 day max
counter=144;while [ $counter -gt 0 ]; do sleep 10m; wget $proxyOptions -N --content-disposition --tries=1 --no-cookies --header "Cookie: $cookies" $urlPrefix"/al1/protected/payload.xhtml"; counter=$(($counter-1)); done &
bdpid=$!

dataFilePaths=("/al1/protected/downloadData/mag/level2/2026/02/15/L2_AL1_MAG_20260215_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/02/14/L2_AL1_MAG_20260214_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/02/13/L2_AL1_MAG_20260213_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/02/12/L2_AL1_MAG_20260212_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/02/11/L2_AL1_MAG_20260211_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/02/10/L2_AL1_MAG_20260210_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/02/09/L2_AL1_MAG_20260209_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/02/08/L2_AL1_MAG_20260208_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/02/07/L2_AL1_MAG_20260207_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/02/06/L2_AL1_MAG_20260206_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/02/05/L2_AL1_MAG_20260205_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/02/04/L2_AL1_MAG_20260204_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/02/03/L2_AL1_MAG_20260203_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/01/15/L2_AL1_MAG_20260115_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/02/02/L2_AL1_MAG_20260202_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/02/01/L2_AL1_MAG_20260201_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/01/31/L2_AL1_MAG_20260131_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/01/30/L2_AL1_MAG_20260130_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/01/29/L2_AL1_MAG_20260129_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/01/28/L2_AL1_MAG_20260128_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/01/27/L2_AL1_MAG_20260127_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/01/26/L2_AL1_MAG_20260126_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/01/25/L2_AL1_MAG_20260125_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/01/24/L2_AL1_MAG_20260124_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/01/23/L2_AL1_MAG_20260123_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/01/22/L2_AL1_MAG_20260122_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/01/21/L2_AL1_MAG_20260121_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/01/20/L2_AL1_MAG_20260120_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/01/19/L2_AL1_MAG_20260119_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/01/18/L2_AL1_MAG_20260118_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/01/17/L2_AL1_MAG_20260117_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2026/01/16/L2_AL1_MAG_20260116_V00.nc?mag" )

i=0;
for file in ${dataFilePaths[@]}
do 
	echo $file; 
	i=$(($i+1));
	wget $proxyOptions -x --max-redirect=0 --content-disposition --tries=1 --no-cookies --header "Cookie: $cookies" $urlPrefix$file;
	if [ $? -ne 0 ]; then
		echo "Error: Limits reached or session expired, terminating without downloading file $i: $file. You may login again later to download script for the new session and resume downloads." 
		kill -9 $bdpid
		exit -1;
	fi
done
echo "Your downloads($i) are complete."

kill -9 $bdpid

