# Clone ARO, touch __init__ to make accessible
git clone https://github.com/mertyg/vision-language-models-are-bows.git densely_captioned_images/repro/eval/ARO
touch densely_captioned_images/repro/eval/ARO/__init__.py

# Clone specific VL-Checklist, as we apply patch
git clone https://github.com/om-ai-lab/VL-CheckList.git densely_captioned_images/repro/eval/VLChecklist
cd densely_captioned_images/repro/eval/VLChecklist
git fetch origin 3bb214e283726f80cb465f2dfd95dc9dfa32cc08
git reset --hard FETCH_HEAD
touch __init__.py
git apply ../../../../patches/vlc-patch.patch
pip install -e .
cd ../../../../

# Clone specific VL-Checklist, as we apply patch
git clone https://github.com/Computer-Vision-in-the-Wild/Elevater_Toolkit_IC.git densely_captioned_images/repro/eval/ElevaterIC
cd densely_captioned_images/repro/eval/ElevaterIC
git fetch origin aae48b1e9c0a4425e2e05030682ff261d684cc4d
git reset --hard FETCH_HEAD
touch __init__.py
git apply ../../../../patches/elevater-patch.patch
pip install -e .
cd ../../../../

# Get localized narratives
git clone https://github.com/google/localized-narratives.git densely_captioned_images/repro/eval/localized_narratives
