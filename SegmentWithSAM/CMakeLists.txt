#-----------------------------------------------------------------------------
set(MODULE_NAME SegmentWithSAM)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  )
set(EXTENSION_HOMEPAGE "https://github.com/mazurowski-lab/SlicerSegmentWithSAM")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Zafer Yildiz (Mazurowski Lab, Duke University)")
set(EXTENSION_DESCRIPTION "SegmentWithSAM aims to asist it