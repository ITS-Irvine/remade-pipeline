for f in `ls output/appliance/hold/OD_Appliances_Metals_*`; do echo "$f "`basename $f`"============="; diff $f output/appliance/`basename $f`; done
