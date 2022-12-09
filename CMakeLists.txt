cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SegmentWithSAM)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/mazurowski-lab/SlicerSegmentWithSAM")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Zafer Yildiz (Mazurowski Lab, Duke University)")
set(EXTENSION_DESCRIPTION "SegmentWithSAM aims to asist its users in segmenting medical data on 3D Slicer by comprehensively integrating the Segment Anything Model (SAM) developed by Meta.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/mazurowski-lab/SlicerSegmentWithSAM/main/SegmentWithSAM/Resources/Icons/SegmentWithSAM.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/mazurowski-lab/SlicerSegmentWithSAM/main/Screenshots/sws1.png https://raw.githubusercontent.com/mazurowski-lab/SlicerSegmentWithSAM/main/Screenshots/sws2.png https://raw.githubusercontent.com/mazurowski-lab/SlicerSegmentWithSAM/main/Screenshots/sws3.png https://raw.githubusercontent.com/mazurowski-lab/SlicerSegmentWithSAM/main