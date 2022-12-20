
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
                    shutil.copyfile(downloadedFilePath, self.checkpointFolder + "sam_vit_b_01ec64.pth")
                    slicer.progressWindow.close()
        
        if not os.path.exists(self.checkpointFolder + "sam2_hiera_tiny.pt"):
            if slicer.util.confirmOkCancelDisplay(
                "Would you like to use SAM-2 (Tiny) checkpoint (148 MB)? Click OK to install it now!"
            ):
                
                slicer.progressWindow = slicer.util.createProgressDialog()
                self.sampleDataLogic = SampleData.SampleDataLogic()
                self.sampleDataLogic.logMessage = self.reportProgress

                checksum = "SHA256:65b50056e05bcb13694174f51bb6da89c894b57b75ccdf0ba6352c597c5d1125"    
                downloadedFilePath = self.sampleDataLogic.downloadFileIntoCache("https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt", "sam2_hiera_tiny.pt", checksum)
                
                if self.sampleDataLogic.downloadPercent and self.sampleDataLogic.downloadPercent == 100:
                    shutil.copyfile(downloadedFilePath, self.checkpointFolder + "sam2_hiera_tiny.pt")
                    slicer.progressWindow.close()

        if not os.path.exists(self.checkpointFolder + "sam2_hiera_small.pt"):
            if slicer.util.confirmOkCancelDisplay(
                "Would you like to use SAM-2 (Small) checkpoint (175 MB)? Click OK to install it now!"
            ):
                
                slicer.progressWindow = slicer.util.createProgressDialog()
                self.sampleDataLogic = SampleData.SampleDataLogic()
                self.sampleDataLogic.logMessage = self.reportProgress

                checksum = "SHA256:95949964d4e548409021d47b22712d5f1abf2564cc0c3c765ba599a24ac7dce3"    
                downloadedFilePath = self.sampleDataLogic.downloadFileIntoCache("https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt", "sam2_hiera_small.pt", checksum)
                
                if self.sampleDataLogic.downloadPercent and self.sampleDataLogic.downloadPercent == 100:
                    shutil.copyfile(downloadedFilePath, self.checkpointFolder + "sam2_hiera_small.pt")
                    slicer.progressWindow.close()

        if not os.path.exists(self.checkpointFolder + "sam2_hiera_base_plus.pt"):
            if slicer.util.confirmOkCancelDisplay(
                "Would you like to use SAM-2 (Base Plus) checkpoint (308 MB)? Click OK to install it now!"
            ):
                
                slicer.progressWindow = slicer.util.createProgressDialog()
                self.sampleDataLogic = SampleData.SampleDataLogic()
                self.sampleDataLogic.logMessage = self.reportProgress

                checksum = "SHA256:d0bb7f236400a49669ffdd1be617959a8b1d1065081789d7bbff88eded3a8071"    
                downloadedFilePath = self.sampleDataLogic.downloadFileIntoCache("https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt", "sam2_hiera_base_plus.pt", checksum)
                
                if self.sampleDataLogic.downloadPercent and self.sampleDataLogic.downloadPercent == 100:
                    shutil.copyfile(downloadedFilePath, self.checkpointFolder + "sam2_hiera_base_plus.pt")
                    slicer.progressWindow.close()

        if not os.path.exists(self.checkpointFolder + "sam2_hiera_large.pt"):
            if slicer.util.confirmOkCancelDisplay(
                "Would you like to use SAM-2 (Base Plus) checkpoint (856 MB)? Click OK to install it now!"
            ):
                
                slicer.progressWindow = slicer.util.createProgressDialog()
                self.sampleDataLogic = SampleData.SampleDataLogic()
                self.sampleDataLogic.logMessage = self.reportProgress

                checksum = "SHA256:7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b"    
                downloadedFilePath = self.sampleDataLogic.downloadFileIntoCache("https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt", "sam2_hiera_large.pt", checksum)
                
                if self.sampleDataLogic.downloadPercent and self.sampleDataLogic.downloadPercent == 100:
                    shutil.copyfile(downloadedFilePath, self.checkpointFolder + "sam2_hiera_large.pt")
                    slicer.progressWindow.close()

        try:
            import git
        except ModuleNotFoundError:
            slicer.util.pip_install("gitpython")

        try: 
            import git
        except ModuleNotFoundError:
            raise RuntimeError("There is a problem about the installation of 'gitpython' package. Please try again to install!")

        if not os.path.exists(self.resourcePath("UI") + "/../../../sam2_configs"):
            copyFolder = self.resourcePath("UI") + "/../../../repo_copy"
            os.makedirs(copyFolder)
            git.Repo.clone_from("https://github.com/mazurowski-lab/SlicerSegmentWithSAM.git", copyFolder)
            shutil.move(copyFolder + "/sam2", self.resourcePath("UI") + "/../../../sam2")
            shutil.move(copyFolder + "/sam2_configs", self.resourcePath("UI") + "/../../../sam2_configs")
            shutil.move(copyFolder + "/setup.py", self.resourcePath("UI") + "/../../../setup.py")
            shutil.rmtree(copyFolder, ignore_errors=True)

        try:
            import PyTorchUtils
        except ModuleNotFoundError:
            extensionName = 'PyTorch'
            em = slicer.app.extensionsManagerModel()
            em.interactive = False  # prevent display of popups
            restart = True
            if not em.installExtensionFromServer(extensionName, restart):
                raise ValueError(f"Failed to install {extensionName} extension")

        minimumTorchVersion = "2.0.0"
        minimumTorchVisionVersion = "0.15.0"
        torchLogic = PyTorchUtils.PyTorchUtilsLogic()

        import platform
        if not torchLogic.torchInstalled():
            if "Windows" in platform.platform() or "Linux" in platform.platform():
                slicer.util.delayDisplay("PyTorch Python package is required. Installing... (it may take several minutes)")
                torch = torchLogic.installTorch(
                    askConfirmation=True,
                    forceComputationBackend="cu117",
                    torchVersionRequirement=f">={minimumTorchVersion}",
                    torchvisionVersionRequirement=f">={minimumTorchVisionVersion}",
                )
                if torch is None:
                    raise ValueError("You need to install PyTorch to use SegmentWithSAM!")
            else:
                slicer.util.delayDisplay("PyTorch Python package is required. Installing... (it may take several minutes)")
                slicer.util.pip_install("torch torchvision torchaudio")
        
        import torch

        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                "'segment-anything' is missing. Click OK to install it now!"
            ):
                slicer.util.pip_install("https://github.com/facebookresearch/segment-anything/archive/6fdee8f2727f4506cfbbe553e23b895e27956588.zip") 
        try: 
            from segment_anything import sam_model_registry, SamPredictor
        except ModuleNotFoundError:
            raise RuntimeError("There is a problem about the installation of 'segment-anything' package. Please try again to install!")
        
        try:
            import hydra
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                "'hydra' is missing. Click OK to install it now!"
            ): 
                slicer.util.pip_install("hydra-core")
        try: 
            import hydra
        except ModuleNotFoundError:
            raise RuntimeError("There is a problem about the installation of 'hydra' package. Please try again to install!")


        try:
            import ninja
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                "'ninja' is missing. Click OK to install it now!"
            ): 
                slicer.util.pip_install("ninja")
        try: 
            import ninja
        except ModuleNotFoundError:
            raise RuntimeError("There is a problem about the installation of 'ninja' package. Please try again to install!")


        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            slicer.util.pip_install("matplotlib")
        
        try: 
            import  matplotlib.pyplot as plt
        except ModuleNotFoundError:
            raise RuntimeError("There is a problem about the installation of 'matplotlib' package. Please try again to install!")

        try:
            import tqdm
        except ModuleNotFoundError:
            slicer.util.pip_install("tqdm")
        try: 
            import tqdm
        except ModuleNotFoundError:
            raise RuntimeError("There is a problem about the installation of 'tqdm' package. Please try again to install!")

        try:
            import cv2
        except ModuleNotFoundError:
            slicer.util.pip_install("opencv-python")

        try: 
            import cv2
        except ModuleNotFoundError:
            raise RuntimeError("There is a problem about the installation of 'open-cv' package. Please try again to install!")

        try: 
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from sam2.build_sam import build_sam2_video_predictor
            #from setup import setup as sam2setup
        except ModuleNotFoundError:
            raise RuntimeError("There is a problem about the installation of 'sam-2' package. Please try again to install!")
        
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("Working on", self.device)

        self.currentlySegmenting = False
        self.featuresAreExtracted = False

    def createFrames(self):
        if not os.path.exists(self.framesFolder):
            os.makedirs(self.framesFolder)

        oldSliceFiles = glob.glob(self.framesFolder + "/*")
        for filename in oldSliceFiles:
            os.remove(filename)
        self.initializeVariables()
        self.initializeSegmentationProcess()

        for sliceIndex in range(0, self.nofSlices):
            sliceImage = self.getSliceBasedOnSliceAccessorDimension(sliceIndex)
            plt.imsave(self.framesFolder + "/" + "0" * (5 - len(str(sliceIndex))) + str(sliceIndex) + ".jpeg", sliceImage, cmap="gray")


    def show_mask(self, mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_points(self, coords, labels, ax, marker_size=200):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

    def propagation(self, toLeft):

        frame_names = [
            p for p in os.listdir(self.framesFolder)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        inference_state = self.videoPredictor.init_state(video_path=self.framesFolder)
        self.videoPredictor.reset_state(inference_state)

        sliceIndicesToPromptPointCoordinations, sliceIndicesToPromptPointLabels = self.getAllPromptPointsAndLabels()

        ann_frame_idx =  self.getIndexOfCurrentSlice()  # the frame index we interact with
        ann_obj_id = self._parameterNode.GetParameter("SAMCurrentSegment") 
        # give a unique id to each object we interact with (it can be any integers)
        
        _, out_obj_ids, out_mask_logits = self.videoPredictor.add_new_points(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=np.array(sliceIndicesToPromptPointCoordinations[ann_frame_idx], dtype=np.float32),
            labels=np.array(sliceIndicesToPromptPointLabels[ann_frame_idx], np.int32),
        )
        
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.videoPredictor.propagate_in_video(inference_state, reverse=toLeft):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        plt.close("all")
        
        if toLeft:
            for out_frame_idx in range(0, self.getIndexOfCurrentSlice() + 1):
                for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                    orig_map=plt.cm.get_cmap('binary') 
                    # reversing the original colormap using reversed() function 
                    reversed_binary = orig_map.reversed() 
                    plt.imsave(self.framesFolder + "/" + "0" * (5 - len(str(out_frame_idx))) + str(out_frame_idx) + "_mask.jpeg", out_mask[0], cmap=reversed_binary)
            
            self.updateSlicesWithSegmentationMasks(0, self.getIndexOfCurrentSlice())
        else:
            for out_frame_idx in range(self.getIndexOfCurrentSlice(), self.nofSlices):
                for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                    orig_map=plt.cm.get_cmap('binary') 
                    # reversing the original colormap using reversed() function 
                    reversed_binary = orig_map.reversed() 
                    plt.imsave(self.framesFolder + "/" + "0" * (5 - len(str(out_frame_idx))) + str(out_frame_idx) + "_mask.jpeg", out_mask[0], cmap=reversed_binary)

            self.updateSlicesWithSegmentationMasks(self.getIndexOfCurrentSlice(), self.nofSlices - 1)

    def getAllPromptPointsAndLabels(self):
        sliceIndicesToPromptPointCoordinations = {}
        sliceIndicesToPromptPointLabels = {}

        nofPositivePromptPoints = self.positivePromptPointsNode.GetNumberOfControlPoints()
        for i in range(nofPositivePromptPoints):
            if self.positivePromptPointsNode.GetNthControlPointVisibility(i):
                pointRAS = [0, 0, 0]
                self.positivePromptPointsNode.GetNthControlPointPositionWorld(i, pointRAS)
                pointIJK = [0, 0, 0, 1]
                self.volumeRasToIjk.MultiplyPoint(np.append(pointRAS, 1.0), pointIJK)
                pointIJK = [int(round(c)) for c in pointIJK[0:3]]

                if self.sliceAccessorDimension == 2:
                    if pointIJK[0] not in sliceIndicesToPromptPointCoordinations.keys():
                        sliceIndicesToPromptPointCoordinations[pointIJK[0]] = []
                        sliceIndicesToPromptPointLabels[pointIJK[0]] = []
                    sliceIndicesToPromptPointCoordinations[pointIJK[0]].append([pointIJK[1], pointIJK[2]])
                    sliceIndicesToPromptPointLabels[pointIJK[0]].append(1)
                elif self.sliceAccessorDimension == 1:
                    if pointIJK[1] not in sliceIndicesToPromptPointCoordinations.keys():
                        sliceIndicesToPromptPointCoordinations[pointIJK[1]] = []
                        sliceIndicesToPromptPointLabels[pointIJK[1]] = []
                    sliceIndicesToPromptPointCoordinations[pointIJK[1]].append([pointIJK[0], pointIJK[2]])
                    sliceIndicesToPromptPointLabels[pointIJK[1]].append(1)

                elif self.sliceAccessorDimension == 0:
                    if pointIJK[2] not in sliceIndicesToPromptPointCoordinations.keys():
                        sliceIndicesToPromptPointCoordinations[pointIJK[2]] = []
                        sliceIndicesToPromptPointLabels[pointIJK[2]] = []
                    sliceIndicesToPromptPointCoordinations[pointIJK[2]].append([pointIJK[0], pointIJK[1]])
                    sliceIndicesToPromptPointLabels[pointIJK[2]].append(1)

        nofNegativePromptPoints = self.negativePromptPointsNode.GetNumberOfControlPoints()
        for i in range(nofNegativePromptPoints):
            if self.negativePromptPointsNode.GetNthControlPointVisibility(i):
                pointRAS = [0, 0, 0]
                self.negativePromptPointsNode.GetNthControlPointPositionWorld(i, pointRAS)
                pointIJK = [0, 0, 0, 1]
                self.volumeRasToIjk.MultiplyPoint(np.append(pointRAS, 1.0), pointIJK)
                pointIJK = [int(round(c)) for c in pointIJK[0:3]]

                if self.sliceAccessorDimension == 2:
                    if pointIJK[0] not in sliceIndicesToPromptPointCoordinations.keys():
                        sliceIndicesToPromptPointCoordinations[pointIJK[0]] = []
                        sliceIndicesToPromptPointLabels[pointIJK[0]] = []
                    sliceIndicesToPromptPointCoordinations[pointIJK[0]].append([pointIJK[1], pointIJK[2]])
                    sliceIndicesToPromptPointLabels[pointIJK[0]].append(0)
                elif self.sliceAccessorDimension == 1:
                    if pointIJK[1] not in sliceIndicesToPromptPointCoordinations.keys():
                        sliceIndicesToPromptPointCoordinations[pointIJK[1]] = []
                        sliceIndicesToPromptPointLabels[pointIJK[1]] = []
                    sliceIndicesToPromptPointCoordinations[pointIJK[1]].append([pointIJK[0], pointIJK[2]])
                    sliceIndicesToPromptPointLabels[pointIJK[1]].append(0)
                elif self.sliceAccessorDimension == 0:
                    if pointIJK[2] not in sliceIndicesToPromptPointCoordinations.keys():
                        sliceIndicesToPromptPointCoordinations[pointIJK[2]] = []
                        sliceIndicesToPromptPointLabels[pointIJK[2]] = []
                    sliceIndicesToPromptPointCoordinations[pointIJK[2]].append([pointIJK[0], pointIJK[1]])
                    sliceIndicesToPromptPointLabels[pointIJK[2]].append(0)

        return sliceIndicesToPromptPointCoordinations, sliceIndicesToPromptPointLabels


    def propagateThroughAllSlices(self):
        with slicer.util.MessageDialog("Propagating through all slices..."):
            with slicer.util.WaitCursor():