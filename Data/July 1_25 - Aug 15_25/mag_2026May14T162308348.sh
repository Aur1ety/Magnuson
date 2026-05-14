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

dataFilePaths=("/al1/protected/downloadData/mag/level2/2025/08/15/L2_AL1_MAG_20250815_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/08/14/L2_AL1_MAG_20250814_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/08/13/L2_AL1_MAG_20250813_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/08/12/L2_AL1_MAG_20250812_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/08/11/L2_AL1_MAG_20250811_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/08/10/L2_AL1_MAG_20250810_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/08/09/L2_AL1_MAG_20250809_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/08/08/L2_AL1_MAG_20250808_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/08/07/L2_AL1_MAG_20250807_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/08/06/L2_AL1_MAG_20250806_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/08/05/L2_AL1_MAG_20250805_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/08/04/L2_AL1_MAG_20250804_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/08/03/L2_AL1_MAG_20250803_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/08/02/L2_AL1_MAG_20250802_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/08/01/L2_AL1_MAG_20250801_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/31/L2_AL1_MAG_20250731_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/30/L2_AL1_MAG_20250730_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/29/L2_AL1_MAG_20250729_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/28/L2_AL1_MAG_20250728_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/27/L2_AL1_MAG_20250727_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/26/L2_AL1_MAG_20250726_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/25/L2_AL1_MAG_20250725_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/24/L2_AL1_MAG_20250724_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/23/L2_AL1_MAG_20250723_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/22/L2_AL1_MAG_20250722_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/21/L2_AL1_MAG_20250721_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/20/L2_AL1_MAG_20250720_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/19/L2_AL1_MAG_20250719_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/18/L2_AL1_MAG_20250718_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/17/L2_AL1_MAG_20250717_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/16/L2_AL1_MAG_20250716_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/15/L2_AL1_MAG_20250715_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/14/L2_AL1_MAG_20250714_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/13/L2_AL1_MAG_20250713_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/12/L2_AL1_MAG_20250712_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/11/L2_AL1_MAG_20250711_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/10/L2_AL1_MAG_20250710_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/09/L2_AL1_MAG_20250709_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/08/L2_AL1_MAG_20250708_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/07/L2_AL1_MAG_20250707_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/06/L2_AL1_MAG_20250706_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/05/L2_AL1_MAG_20250705_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/04/L2_AL1_MAG_20250704_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/03/L2_AL1_MAG_20250703_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/02/L2_AL1_MAG_20250702_V00.nc?mag" "/al1/protected/downloadData/mag/level2/2025/07/01/L2_AL1_MAG_20250701_V00.nc?mag" )

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

