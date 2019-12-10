for i in $1/* ; do
	if [ -d $i ] ; then
		FN=BOLD.nii
		if [ -f $i/BOLD.nii.gz ] ; then
			FN=BOLD.nii.gz
		fi
		python utils/stackNiftis.py -d $i/dag/ -o $i/corrected_dag.nii.gz -c $i/$FN
	fi
done
