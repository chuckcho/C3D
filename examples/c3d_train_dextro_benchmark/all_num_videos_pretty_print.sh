jv all_num_videos_2016_02_05.json \
  | egrep ":" \
  | sed -e 's/,$//' -e's/^ *//' -e's/"//g' \
  | sort -n --field-separator=: -k2|more \
  > all_num_videos_sorted_2016_02_05.txt
