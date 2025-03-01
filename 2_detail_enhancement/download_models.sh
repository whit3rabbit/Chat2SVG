wget -O sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

wget -O aamXLAnimeMix_v10.safetensors https://civitai.com/api/download/models/303526?type=Model&format=SafeTensor&size=full&fp=fp16

mkdir -p models
mv sam_vit_h_4b8939.pth models/
mv aamXLAnimeMix_v10.safetensors models/
