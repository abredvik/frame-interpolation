URL="https://vision.middlebury.edu/stereo/data/scenes2014/datasets"

download () {
	mkdir -p $1
	wget -nc -nv -P $1 $URL/${1^}-imperfect/im0.png
	wget -nc -nv -P $1 $URL/${1^}-imperfect/im1.png
	wget -nc -nv -P $1 $URL/${1^}-imperfect/disp0.pfm
	wget -nc -nv -P $1 $URL/${1^}-imperfect/disp1.pfm
}

mkdir -p images
cd images

if [ $# -eq 0 ]
then
	download "adirondack"
	download "flowers"
	download "jadeplant"
else
	for dataset in $@
	do
		download $dataset
	done
fi

