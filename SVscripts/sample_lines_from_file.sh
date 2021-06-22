awk -v n=109977801 -v p=1000000 '
  BEGIN {srand()}
  rand() * n-- < p {p--; print}' < ./cleaned/sv.json 
