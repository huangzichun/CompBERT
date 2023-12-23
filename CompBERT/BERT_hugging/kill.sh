l=$(ps -aux | grep "${1}")
echo "${l}"
ids=$(echo "${l}" | tr -s " " | cut -d' ' -f2)
echo "${ids}"
for i in ${ids}
do
  echo "killing ${i}"
  $(kill -9 "${i}")
done
