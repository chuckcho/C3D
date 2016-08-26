videodir=/media/6TB2/Videos/test-streams
#ls ${videodir}/*.mp4 | xargs -I {} ffmpeg -i {} -vcodec copy -acodec copy -f null /dev/null 2>&1 | grep 'frame=' | awk '{print $2}'
inputfile=/home/chuck/projects/C3D/examples/c3d_feature_extraction/prototxt/testvideos_input.txt
outputfile=/home/chuck/projects/C3D/examples/c3d_feature_extraction/prototxt/testvideos_output.txt
outdir=${videodir}/C3D_features

numframesc3d=16
minframenum=16

rm -f "${inputfile}" "${outputfile}"
rm -rf "${outdir}"/*
mkdir -p "${outdir}"

FILES=${videodir}/*.mp4
for f in $FILES
do
  basef="${f##*/}"
  basefnoext="${basef%.*}"
  fnoext=${videodir}/${basefnoext}
  echo "Processing f=\"${f}\", basef=\"${basef}\"..."
  framenum=$(ffmpeg -i "${f}" -vcodec copy -acodec copy -f null /dev/null 2>&1 | grep 'frame=' | sed -e 's/frame=[ ]*/frame=/' | awk '{print $1}' | sed -e 's/frame=//' )
  echo "#frame=${framenum}"

  if [ -z ${framenum} ]; then
    echo "[error] framenum was not extracted correctly!"
    continue
  fi

  if [ ${framenum} -lt ${minframenum} ]; then
    echo "too few frames. Skipping this shot..."
    continue
  fi

  midfrmnum=$((framenum / 2 - 8))
  echo "\"${fnoext}\" $(( 10#${midfrmnum} + 1 )) 0"
  echo "${outdir}/${basefnoext}_$(( 10#${midfrmnum} + 1 ))"

  echo "${fnoext} $(( 10#${midfrmnum} + 1 )) 0" >> "${inputfile}"
  echo "${outdir}/${basefnoext}_$(( 10#${midfrmnum} + 1 ))" >> "${outputfile}"

  #echo mkdir -p "${outdir}/${basefnoext}"
  #mkdir -p "${outdir}/${basefnoext}"

done
