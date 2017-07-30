#!/usr/bin/env bash
#
# purpose:
# this script defines popular functions
#

# executes command CMD if required REQ_VER is greater than CUR_VER
function meets_version() {
	REQ_VER=$1;
	CUR_VER=$2;
	CMD=${@:3:$#-2};
	if [ -z $CUR_VER ] || [ "$(printf "$REQ_VER\n$CUR_VER" | sort -V | head -n1)" == "$CUR_VER" ] && [ "$CUR_VER" != "$REQ_VER" ];
	then
		eval "$CMD";
	else
		echo "requirement: $REQ_VER met. current version: $CUR_VER";
	fi
}
