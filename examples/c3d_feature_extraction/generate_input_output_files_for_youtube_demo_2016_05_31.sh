#!/usr/bin/env bash

youtubedir=/media/6TB/Videos/youtube-dog-videos-for-demo-2016-Jun
#ls ${youtubedir}/*.avi | xargs -I {} ffmpeg -i {} -vcodec copy -acodec copy -f null /dev/null 2>&1 | grep 'frame=' | awk '{print $2}'
inputfile=/home/chuck/projects/C3D/examples/c3d_feature_extraction/prototxt/youtube_demo_input_list_video.txt
outputfile=/home/chuck/projects/C3D/examples/c3d_feature_extraction/prototxt/youtube_demo_output_list_video_prefix.txt
outdir=${youtubedir}/C3D_features

numframesc3d=16

minframenum=30

rm -f $inputfile $outputfile
rm -rf $outdir/*

FILES=${youtubedir}/*.mp4
for f in $FILES
do
  basef="${f##*/}"
  basefnoext="${basef%.*}"
  fnoext=${youtubedir}/${basefnoext}
  echo "Processing f=\"${f}\", basef=\"${basef}\"..."
  framenum=$(ffmpeg -i "${f}" -vcodec copy -acodec copy -f null /dev/null 2>&1 | grep 'frame=' | sed -e 's/frame= /frame=/' | awk '{print $1}' | sed -e 's/frame=//' )
  echo "#frame=${framenum}"

  if [ $framenum -le $minframenum ]; then
    echo "too few frames. Skipping this shot..."
    continue
  fi;

  curframe=0
  for i in $(seq -f "%05g" 0 $numframesc3d $((framenum - minframenum + 1)) )
  #for i in $(seq -f "%05g" 0 $numframesc3d $framenum)
  do
    echo "\"${fnoext}\" $(( 10#$i + 1 )) 0"
    echo "$outdir/$basefnoext/$(( 10#$i + 1 ))"

    echo "${fnoext} $(( 10#$i + 1 )) 0" >> "${inputfile}"
    echo "$outdir/$basefnoext/$(( 10#$i + 1 ))" >> "${outputfile}"

  done

  echo mkdir -p "${outdir}/${basefnoext}"
  mkdir -p "${outdir}/${basefnoext}"

  #while [[ $curframe -le $framenum ]]
  #do
  #  echo "curframe=${curframe}"
  #  curframe=$(( $curframe + $numframesc3d ))
  #  echo $outdir/$f/
  #done

done
