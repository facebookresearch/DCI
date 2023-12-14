#!/bin/bash

# --------------------------------------------------------
# Download images for HAKE Dataset.
# --------------------------------------------------------
# TODO all drive links should have checksums

if [! -d data/hake ];then
    mkdir data/hake
fi
if [! -d data/hake/images ];then
    mkdir data/hake/images
fi
cd data/hake

# ---------------V-COCO Dataset--------------------
echo "Downloading V-COCO Dataset"

URL_2017_Train_images=http://images.cocodataset.org/zips/train2017.zip
URL_2017_Val_images=http://images.cocodataset.org/zips/val2017.zip
#URL_2017_Test_images=http://images.cocodataset.org/zips/test2017.zip

wget -N $URL_2017_Train_images
wget -N $URL_2017_Val_images
#wget -N $URL_2017_Test_images

if [! -d vcoco ];then
    mkdir vcoco
fi

unzip -q train2017.zip -d vcoco/
unzip -q val2017.zip -d vcoco/
#unzip test2017.zip -d vcoco/

rm train2017.zip
rm val2017.zip
#rm test2017.zip

echo "V-COCO Dataset Downloaded!\n"

# ---------------HICO-DET Dataset-------------------
echo "Downloading HICO-DET Dataset"

# source: https://github.com/YueLiao/CDN#hico-det
python ../reproduction/densely_captioned_images/repro/setup_data/download_from_drive.py '1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk' hico_20160224_det.tar.gz
tar -xzf hico_20160224_det.tar.gz -C ./
rm hico_20160224_det.tar.gz

echo "HICO-DET Dataset Downloaded!\n"


# ---------------hcvrd Dataset(visual genome)-------
echo "Downloading HCVRD(part) Dataset"
if [! -d hcvrd ];then
    mkdir hcvrd
fi
# source: https://github.com/DirtyHarryLYL/HAKE/blob/master/Images/download_image/hcvrd_url.json
python ../reproduction/densely_captioned_images/repro/setup_data/download.py '../reproduction/densely_captioned_images/repro/setup_data/hcvrd_url.json' ./hcvrd

echo "HCVRD(part) Dataset Downloaded!\n"


# ---------------openimages Dataset------------------

# Must run the download_from_drive setup...
echo "Downloading openimages(part) Dataset"

# source: https://github.com/DirtyHarryLYL/HAKE/tree/master/Images#how-to-download-images
python ../reproduction/densely_captioned_images/repro/setup_data/download_from_drive.py '1XTWYLyL1h-9jJ49dsXmtRCv8GcupVrvM' openimages.tar.gz
tar -xzf openimages.tar.gz -C ./
rm openimages.tar.gz

echo "openimages(part) Dataset Downloaded!\n"


# ---------------pic Dataset-------------------------

echo "Downloading pic Dataset"

# source: https://picdataset.com/challenge/task/download/
python ../reproduction/densely_captioned_images/repro/setup_data/download_from_drive.py '1fBJh0mdWhOkOyN5X8is7a2MDb2CE7eCw' pic.tar.gz
tar -xzf pic.tar.gz -C ./
rm pic.tar.gz
mkdir pic
mv image/val/* pic
mv image/train/* pic
rm -rf image

echo "pic Dataset Downloaded!\n"


# ---------------hake uploads-------------------------

# Sources: https://github.com/DirtyHarryLYL/HAKE/tree/master/Images#how-to-download-images
echo "Downloading hake Dataset 1"

python ../reproduction/densely_captioned_images/repro/setup_data/download_from_drive.py '1Smrsy9AsOUyvj66ytGmB5M3WknljwuXL' hake_images_20190730.tar.gz
tar -xzf hake_images_20190730.tar.gz -C ./
rm hake_images_20190730.tar.gz

echo "hake part 1 Dataset Downloaded!\n"


echo "Downloading hake Dataset 2"

python ../reproduction/densely_captioned_images/repro/setup_data/download_from_drive.py '14K_4FfjviJNDVLJdGM96W2ZLN55dDb2-' hake_images_20200614.tar.gz
tar -xzf hake_images_20200614.tar.gz -C ./
rm hake_images_20200614.tar.gz

echo "hake part 2 Dataset Downloaded!\n"


# ---------------SWiG-------------------------
echo "Setting up SWiG"
cd ..
mkdir swig

# source: https://github.com/om-ai-lab/VL-CheckList/blob/main/DATASETS.md#swig
wget -N  https://swig-data-weights.s3.us-east-2.amazonaws.com/images_512.zip
unzip -q images_512.zip -d ./
rm images_512.zip
mv images_512 swig


echo "SWiG Downloaded!\n"

# ---------------VG-------------------------
echo "Setting up Visual Genome"

mkdir vg
mkdir vg/VG_100K
mkdir vg/VG_100K_2

# source: https://github.com/om-ai-lab/VL-CheckList/blob/main/DATASETS.md#vg--vaw
wget -N https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget -N https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip

unzip -q images.zip -d vg/
unzip -q images2.zip -d vg/
rm images.zip
rm images2.zip

echo "Visual Genome Downloaded!\n"

# ----- clear any files that didn't download correctly -----

echo "Removing improperly downloaded/missing files"
find . -size 0 -type f -name "*.jpg" -print -delete
echo "Files processed, remaining are VL-Checklist-usable"