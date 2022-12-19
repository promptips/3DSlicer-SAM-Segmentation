
import glob
import os
import pickle
import numpy as np
import qt
import vtk
import shutil
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)
from slicer.util import VTKObservationMixin
import SampleData
from PIL import Image

#
# SegmentWithSAM
#

class SegmentWithSAM(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "SegmentWithSAM"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["Zafer Yildiz (Mazurowski Lab, Duke University)"]
        self.parent.helpText = """
The SegmentWithSAM module aims to assist its users in segmenting medical data by integrating
the <a href="https://github.com/facebookresearch/segment-anything">Segment Anything Model (SAM)</a>
developed by Meta.<br>
<br>
See more information in <a href="https://github.com/mazurowski-lab/SlicerSegmentWithSAM">module documentation</a>.
"""
        self.parent.acknowledgementText = """
This file was originally developed by Zafer Yildiz (Mazurowski Lab, Duke University).
"""


#
# SegmentWithSAMWidget
#


class SegmentWithSAMWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        global sam_model_registry
        global SamPredictor
        global torch
        global cv2
        global hydra
        global tqdm
        global build_sam2
        global SAM2ImagePredictor
        global build_sam2_video_predictor
        global sam2_setup
        global setuptools
        global ninja
        global plt
        global git

        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False
        self.slicesFolder = self.resourcePath("UI") + "/../../../slices"
        self.featuresFolder = self.resourcePath("UI") + "/../../../features"
        self.framesFolder = self.resourcePath("UI") + "/../../../frames"

        self.modelVersion = "vit_h"
        self.checkpointName = "sam_vit_h_4b8939.pth"
        self.checkpointFolder = self.resourcePath("UI") + "/../../../model_checkpoints/"
        self.modelCheckpoint = self.checkpointFolder + self.checkpointName
        self.masks = None
        self.mask_threshold = 0

        vtk.vtkObject.GlobalWarningDisplayOff()

        import shutil

        if not os.path.exists(self.checkpointFolder):
            os.makedirs(self.checkpointFolder)

        if not os.path.exists(self.checkpointFolder + "sam_vit_h_4b8939.pth"):
            if slicer.util.confirmOkCancelDisplay(
                "Would you like to use SAM (ViT-H) checkpoint (2.38 GB)? Click OK to install it now!"
            ):
                
                slicer.progressWindow = slicer.util.createProgressDialog()
                self.sampleDataLogic = SampleData.SampleDataLogic()
                self.sampleDataLogic.logMessage = self.reportProgress

                checksum = "SHA256:a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e"
                downloadedFilePath = self.sampleDataLogic.downloadFileIntoCache("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", "sam_vit_h_4b8939.pth", checksum)
                
                if self.sampleDataLogic.downloadPercent and self.sampleDataLogic.downloadPercent == 100:
                    shutil.copyfile(downloadedFilePath, self.checkpointFolder + "sam_vit_h_4b8939.pth")
                    slicer.progressWindow.close()

        if not os.path.exists(self.checkpointFolder + "sam_vit_l_0b3195.pth"):
            if slicer.util.confirmOkCancelDisplay(
                "Would you like to use SAM (ViT-L) checkpoint (1.16 GB)? Click OK to install it now!"
            ):
                
                slicer.progressWindow = slicer.util.createProgressDialog()
                self.sampleDataLogic = SampleData.SampleDataLogic()
                self.sampleDataLogic.logMessage = self.reportProgress

                checksum = "SHA256:3adcc4315b642a4d2101128f611684e8734c41232a17c648ed1693702a49a622"
                downloadedFilePath = self.sampleDataLogic.downloadFileIntoCache("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", "sam_vit_l_0b3195.pth", checksum)
                
                if self.sampleDataLogic.downloadPercent and self.sampleDataLogic.downloadPercent == 100:
                    shutil.copyfile(downloadedFilePath, self.checkpointFolder + "sam_vit_l_0b3195.pth")
                    slicer.progressWindow.close()

        if not os.path.exists(self.checkpointFolder + "sam_vit_b_01ec64.pth"):
            if slicer.util.confirmOkCancelDisplay(
                "Would you like to use SAM (ViT-B) checkpoint (357 MB)? Click OK to install it now!"
            ):
                
                slicer.progressWindow = slicer.util.createProgressDialog()
                self.sampleDataLogic = SampleData.SampleDataLogic()
                self.sampleDataLogic.logMessage = self.reportProgress

                checksum = "SHA256:ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"
                downloadedFilePath = self.sampleDataLogic.downloadFileIntoCache("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", "sam_vit_b_01ec64.pth", checksum)
                
                if self.sampleDataLogic.downloadPercent and self.sampleDataLogic.downloadPercent == 100: