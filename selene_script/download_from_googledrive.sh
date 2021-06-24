# when logged in on selene, cd into the directory which you have write access to 
# for example my directory with read-write access: /lustre/fsw/swdl/zcharpy
# Assuming File URL is https://drive.google.com/file/d/1km-0XM4D7HDuSxI2eYigVeYtoQoevlKb/view?usp=sharing
fileId=1km-0XM4D7HDuSxI2eYigVeYtoQoevlKb
fileName=SV_GPT3_56kvocab_CC100Sprakbank_text_document.bin
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)" 
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName}