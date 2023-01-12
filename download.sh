URL="https://vision.middlebury.edu/stereo/data/scenes2014/datasets"

download () {
	mkdir $1
	wget -nv -P $1 $URL/${1^}-imperfect/im0.png
	wget -nv -P $1 $URL/${1^}-imperfect/im1.png
	wget -nv -P $1 $URL/${1^}-imperfect/disp0.pfm
	wget -nv -P $1 $URL/${1^}-imperfect/disp1.pfm
}

mkdir images
cd images

download "adirondack"
download "jadeplant"
download "motorcycle"
